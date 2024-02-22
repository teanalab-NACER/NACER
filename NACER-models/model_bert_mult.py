import torch
import torch.nn as nn

from model_cosine import ModelCosine
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import pipeline

class ModelBertMult(nn.Module):
    """The model for the multiplicative, additive, dot-product interaction functions"""

    def __init__(self, interaction: str, same_w: bool = True, question_emb_dim: int = 4096, entity_emb_dim: int = 300,
                 interaction_dim: int = 512, num_hidden: list = None):
        """Initialize the model object.

        :param interaction: interaction function to use (multiplicative:'mult' or additive:'add')
        :param same_w: use the same matrix W for all features
        :param question_emb_dim: dimension of embedding for questions
        :param entity_emb_dim: dimension of embedding for entities, categories, and literals
        :param interaction_dim: dimensionality for additive interaction
        :param num_hidden: list with sizes of each hidden layer (at least one)
        """
        super(ModelBertMult, self).__init__()
        if interaction not in ['mult', 'add', 'dot']:
            raise ValueError('Invalid interaction function selected.')
        self.interaction = interaction
        self.same_w = same_w
        self.question_emb_dim = question_emb_dim
        self.entity_emb_dim = entity_emb_dim
        if interaction == 'mult':
            if same_w:
                self.W_p = self.W_lit = self.W_cat = self.W_ent = \
                    nn.Linear(question_emb_dim, entity_emb_dim, bias=False)
            else:
                self.W_p = nn.Linear(question_emb_dim, entity_emb_dim, bias=False)
                self.W_lit = nn.Linear(question_emb_dim, entity_emb_dim, bias=False)
                self.W_cat = nn.Linear(question_emb_dim, entity_emb_dim, bias=False)
                self.W_ent = nn.Linear(question_emb_dim, entity_emb_dim, bias=False)
            self.W_s = nn.Linear(entity_emb_dim, entity_emb_dim, bias=False)
        elif interaction == 'add':
            self.interaction_dim = interaction_dim
            if same_w:
                self.Wq_p = self.Wq_lit = self.Wq_cat = self.Wq_ent = \
                    nn.Linear(question_emb_dim, interaction_dim, bias=False)
                self.Ws_p = self.Ws_lit = self.Ws_cat = self.Ws_ent = \
                    nn.Linear(entity_emb_dim, interaction_dim, bias=False)
                self.v_p = self.v_lit = self.v_cat = self.v_ent = nn.Linear(interaction_dim, 1, bias=False)
            else:
                self.Wq_p = nn.Linear(question_emb_dim, interaction_dim, bias=False)
                self.Ws_p = nn.Linear(entity_emb_dim, interaction_dim, bias=False)
                self.Wq_lit = nn.Linear(question_emb_dim, interaction_dim, bias=False)
                self.Ws_lit = nn.Linear(entity_emb_dim, interaction_dim, bias=False)
                self.Wq_cat = nn.Linear(question_emb_dim, interaction_dim, bias=False)
                self.Ws_cat = nn.Linear(entity_emb_dim, interaction_dim, bias=False)
                self.Wq_ent = nn.Linear(question_emb_dim, interaction_dim, bias=False)
                self.Ws_ent = nn.Linear(entity_emb_dim, interaction_dim, bias=False)
                self.v_p = nn.Linear(interaction_dim, 1, bias=False)
                self.v_lit = nn.Linear(interaction_dim, 1, bias=False)
                self.v_cat = nn.Linear(interaction_dim, 1, bias=False)
                self.v_ent = nn.Linear(interaction_dim, 1, bias=False)
            self.Wq_s = nn.Linear(entity_emb_dim, interaction_dim, bias=False)
            self.Ws_s = nn.Linear(entity_emb_dim, interaction_dim, bias=False)
            self.v_s = nn.Linear(interaction_dim, 1, bias=False)

        self.model_cosine = ModelCosine(num_hidden=num_hidden)

        self.BERT = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
        
        for param in self.BERT.parameters():
            param.requires_grad = False
            
        self.W_BERT = nn.Linear(768, question_emb_dim, bias=False)

    def mult_interaction(self, q: torch.Tensor, weights: nn.Linear, s: torch.Tensor):
        """Calculate multiplicative interaction qT x W x s

        :param q: 2D tensor of shape 1 x question_emb_dim
        :param weights: linear layer for W, entity_emb_dim -> question_emb_dim
        :param s: 3D tensor of shape 1 x 'no. of candidate entities' x entity_emb_dim
        :return 3D tensor of shape 1 x 'no. of candidate entities' x 1
        """
        qT_W = weights(q)  # product qT x W; shape: 1 x entity_emb_dim
        elem_mult = qT_W * s  # shape: 1 x no. of candidate entities x entity_emb_dim
        # assert elem_mult.shape == (1, s.shape[1], self.entity_emb_dim)
        return torch.sum(elem_mult, -1, keepdim=True)

    def add_interaction(self, q: torch.Tensor, s: torch.Tensor, weights_q: nn.Linear, weights_s: nn.Linear,
                        v: nn.Linear):
        """Calculate multiplicative interaction vT x tanh(Wq x q + Ws x s)

        :param q: 2D tensor of shape 1 x question_emb_dim
        :param s: 3D tensor of shape 1 x 'no. of candidate entities' x entity_emb_dim
        :param weights_q: linear layer for Wq, question_emb_dim -> interaction_dim
        :param weights_s: linear layer for Ws, entity_emb_dim -> interaction_dim
        :param v: linear layer, interaction_dim -> 1
        :return: 3D tensor of shape 1 x 'no. of candidate entities' x 1
        """
        return v(torch.tanh(weights_q(q) + weights_s(s)))

    def dot_interaction(self, q: torch.Tensor, s: torch.Tensor):
        """Calculate dot-product interaction vT x q

        :param q: 2D tensor of shape 1 x entity_emb_dim
        :param s: 3D tensor of shape 1 x 'no. of candidate entities' x entity_emb_dim
        :return: 3D tensor of shape 1 x 'no. of candidate entities' x 1
        """
        return torch.sum(q * s, -1, keepdim=True)

    def forward(self, overlap_features: torch.Tensor, p_inputs: torch.Tensor, lit_inputs: torch.Tensor,
                cat_inputs: torch.Tensor, ent_inputs: torch.Tensor, s_inputs: torch.Tensor,
                kewer_question_embedding: torch.Tensor, bert_question_embedding: torch.Tensor, 
                previous_answer_embedding: torch.Tensor, features_mask: torch.Tensor = None) -> torch.Tensor:
        """Calculate logit scores for each candidate entity.

        :param overlap_features: 3D tensor of shape 1 x 'no. of candidate entities' x 5
        :param p_inputs: 3D tensor of shape 1 x 'no. of candidate entities' x entity_emb_dim
        :param lit_inputs: 3D tensor of shape 1 x 'no. of candidate entities' x entity_emb_dim
        :param cat_inputs: 3D tensor of shape 1 x 'no. of candidate entities' x entity_emb_dim
        :param ent_inputs: 3D tensor of shape 1 x 'no. of candidate entities' x entity_emb_dim
        :param s_inputs: 3D tensor of shape 1 x 'no. of candidate entities' x entity_emb_dim
        :param question_embedding: 2D tensor of shape 1 x question_emb_dim
        :param previous_answer_embedding: 2D tensor of shape 1 x entity_emb_dim
        :param features_mask: 1D tensor of length num_features
        :return: 2D tensor of shape 1 x 'no. of candidate_entities'
        """
        
        bert_question_embedding = self.W_BERT(bert_question_embedding)
        question_embedding = kewer_question_embedding + bert_question_embedding 
        # question_embedding = question_embedding / torch.linalg.norm(question_embedding)
        
        if self.interaction == 'mult':
            p_interaction = self.mult_interaction(question_embedding, self.W_p,
                                                  p_inputs)  # shape: 1 x no. of candidate entities x 1
            lit_interaction = self.mult_interaction(question_embedding, self.W_lit,
                                                    lit_inputs)  # shape: 1 x no. of candidate entities x 1
            cat_interaction = self.mult_interaction(question_embedding, self.W_cat,
                                                    cat_inputs)  # shape: 1 x no. of candidate entities x 1
            ent_interaction = self.mult_interaction(question_embedding, self.W_ent,
                                                    ent_inputs)  # shape: 1 x no. of candidate entities x 1
            s_interaction = self.mult_interaction(previous_answer_embedding, self.W_s,
                                                  s_inputs)  # shape: 1 x no. of candidate entities x 1
        elif self.interaction == 'add':
            p_interaction = self.add_interaction(question_embedding, p_inputs, self.Wq_p, self.Ws_p, self.v_p)
            lit_interaction = self.add_interaction(question_embedding, lit_inputs, self.Wq_lit, self.Ws_lit, self.v_lit)
            cat_interaction = self.add_interaction(question_embedding, cat_inputs, self.Wq_cat, self.Ws_cat, self.v_cat)
            ent_interaction = self.add_interaction(question_embedding, ent_inputs, self.Wq_ent, self.Ws_ent, self.v_ent)
            s_interaction = self.add_interaction(previous_answer_embedding, s_inputs, self.Wq_s, self.Ws_s, self.v_s)
        elif self.interaction == 'dot':
            p_interaction = self.dot_interaction(question_embedding,
                                                 p_inputs)  # shape: 1 x no. of candidate entities x 1
            lit_interaction = self.dot_interaction(question_embedding,
                                                   lit_inputs)  # shape: 1 x no. of candidate entities x 1
            cat_interaction = self.dot_interaction(question_embedding,
                                                   cat_inputs)  # shape: 1 x no. of candidate entities x 1
            ent_interaction = self.dot_interaction(question_embedding,
                                                   ent_inputs)  # shape: 1 x no. of candidate entities x 1
            s_interaction = self.dot_interaction(previous_answer_embedding,
                                                 s_inputs)  # shape: 1 x no. of candidate entities x 1
        entity_features = torch.cat((overlap_features, p_interaction, lit_interaction, cat_interaction, ent_interaction,
                                     s_interaction), -1)  # shape: 1 x no. of candidate entities x 10
        # assert entity_features.shape == (1, overlap_features.shape[1], 10)
        entity_scores = self.model_cosine(entity_features, features_mask)
        # assert entity_scores.shape == (1, overlap_features.shape[1])
        return entity_scores


class ModelBertMultCAT(nn.Module):
    """The model for the multiplicative, additive, dot-product interaction functions"""

    def __init__(self, interaction: str, same_w: bool = True, question_emb_dim: int = 4096, entity_emb_dim: int = 300,
                 interaction_dim: int = 512, num_hidden: list = None):
        """Initialize the model object.

        :param interaction: interaction function to use (multiplicative:'mult' or additive:'add')
        :param same_w: use the same matrix W for all features
        :param question_emb_dim: dimension of embedding for questions
        :param entity_emb_dim: dimension of embedding for entities, categories, and literals
        :param interaction_dim: dimensionality for additive interaction
        :param num_hidden: list with sizes of each hidden layer (at least one)
        """
        super(ModelBertMultCAT, self).__init__()
        if interaction not in ['mult', 'add', 'dot']:
            raise ValueError('Invalid interaction function selected.')
        self.interaction = interaction
        self.same_w = same_w
        self.question_emb_dim = question_emb_dim
        self.entity_emb_dim = entity_emb_dim
        if interaction == 'mult':
            if same_w:
                self.W_p = self.W_lit = self.W_cat = self.W_ent = \
                    nn.Linear(question_emb_dim, entity_emb_dim, bias=False)
            else:
                self.W_p = nn.Linear(question_emb_dim, entity_emb_dim, bias=False)
                self.W_lit = nn.Linear(question_emb_dim, entity_emb_dim, bias=False)
                self.W_cat = nn.Linear(question_emb_dim, entity_emb_dim, bias=False)
                self.W_ent = nn.Linear(question_emb_dim, entity_emb_dim, bias=False)
            self.W_s = nn.Linear(entity_emb_dim, entity_emb_dim, bias=False)
        elif interaction == 'add':
            self.interaction_dim = interaction_dim
            if same_w:
                self.Wq_p = self.Wq_lit = self.Wq_cat = self.Wq_ent = \
                    nn.Linear(question_emb_dim, interaction_dim, bias=False)
                self.Ws_p = self.Ws_lit = self.Ws_cat = self.Ws_ent = \
                    nn.Linear(entity_emb_dim, interaction_dim, bias=False)
                self.v_p = self.v_lit = self.v_cat = self.v_ent = nn.Linear(interaction_dim, 1, bias=False)
            else:
                self.Wq_p = nn.Linear(question_emb_dim, interaction_dim, bias=False)
                self.Ws_p = nn.Linear(entity_emb_dim, interaction_dim, bias=False)
                self.Wq_lit = nn.Linear(question_emb_dim, interaction_dim, bias=False)
                self.Ws_lit = nn.Linear(entity_emb_dim, interaction_dim, bias=False)
                self.Wq_cat = nn.Linear(question_emb_dim, interaction_dim, bias=False)
                self.Ws_cat = nn.Linear(entity_emb_dim, interaction_dim, bias=False)
                self.Wq_ent = nn.Linear(question_emb_dim, interaction_dim, bias=False)
                self.Ws_ent = nn.Linear(entity_emb_dim, interaction_dim, bias=False)
                self.v_p = nn.Linear(interaction_dim, 1, bias=False)
                self.v_lit = nn.Linear(interaction_dim, 1, bias=False)
                self.v_cat = nn.Linear(interaction_dim, 1, bias=False)
                self.v_ent = nn.Linear(interaction_dim, 1, bias=False)
            self.Wq_s = nn.Linear(entity_emb_dim, interaction_dim, bias=False)
            self.Ws_s = nn.Linear(entity_emb_dim, interaction_dim, bias=False)
            self.v_s = nn.Linear(interaction_dim, 1, bias=False)

        self.model_cosine = ModelCosine(num_hidden=num_hidden)

        self.BERT = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
        
        for param in self.BERT.parameters():
            param.requires_grad = False
            
        self.W_BERT = nn.Linear(768, question_emb_dim, bias=False)
        self.W_CAT = nn.Linear(question_emb_dim*2, question_emb_dim, bias=False)
        

    def mult_interaction(self, q: torch.Tensor, weights: nn.Linear, s: torch.Tensor):
        """Calculate multiplicative interaction qT x W x s

        :param q: 2D tensor of shape 1 x question_emb_dim
        :param weights: linear layer for W, entity_emb_dim -> question_emb_dim
        :param s: 3D tensor of shape 1 x 'no. of candidate entities' x entity_emb_dim
        :return 3D tensor of shape 1 x 'no. of candidate entities' x 1
        """
        qT_W = weights(q)  # product qT x W; shape: 1 x entity_emb_dim
        elem_mult = qT_W * s  # shape: 1 x no. of candidate entities x entity_emb_dim
        # assert elem_mult.shape == (1, s.shape[1], self.entity_emb_dim)
        return torch.sum(elem_mult, -1, keepdim=True)

    def add_interaction(self, q: torch.Tensor, s: torch.Tensor, weights_q: nn.Linear, weights_s: nn.Linear,
                        v: nn.Linear):
        """Calculate multiplicative interaction vT x tanh(Wq x q + Ws x s)

        :param q: 2D tensor of shape 1 x question_emb_dim
        :param s: 3D tensor of shape 1 x 'no. of candidate entities' x entity_emb_dim
        :param weights_q: linear layer for Wq, question_emb_dim -> interaction_dim
        :param weights_s: linear layer for Ws, entity_emb_dim -> interaction_dim
        :param v: linear layer, interaction_dim -> 1
        :return: 3D tensor of shape 1 x 'no. of candidate entities' x 1
        """
        return v(torch.tanh(weights_q(q) + weights_s(s)))

    def dot_interaction(self, q: torch.Tensor, s: torch.Tensor):
        """Calculate dot-product interaction vT x q

        :param q: 2D tensor of shape 1 x entity_emb_dim
        :param s: 3D tensor of shape 1 x 'no. of candidate entities' x entity_emb_dim
        :return: 3D tensor of shape 1 x 'no. of candidate entities' x 1
        """
        return torch.sum(q * s, -1, keepdim=True)

    def forward(self, overlap_features: torch.Tensor, p_inputs: torch.Tensor, lit_inputs: torch.Tensor,
                cat_inputs: torch.Tensor, ent_inputs: torch.Tensor, s_inputs: torch.Tensor,
                kewer_question_embedding: torch.Tensor, bert_question_embedding: torch.Tensor, 
                previous_answer_embedding: torch.Tensor, features_mask: torch.Tensor = None) -> torch.Tensor:
        """Calculate logit scores for each candidate entity.

        :param overlap_features: 3D tensor of shape 1 x 'no. of candidate entities' x 5
        :param p_inputs: 3D tensor of shape 1 x 'no. of candidate entities' x entity_emb_dim
        :param lit_inputs: 3D tensor of shape 1 x 'no. of candidate entities' x entity_emb_dim
        :param cat_inputs: 3D tensor of shape 1 x 'no. of candidate entities' x entity_emb_dim
        :param ent_inputs: 3D tensor of shape 1 x 'no. of candidate entities' x entity_emb_dim
        :param s_inputs: 3D tensor of shape 1 x 'no. of candidate entities' x entity_emb_dim
        :param question_embedding: 2D tensor of shape 1 x question_emb_dim
        :param previous_answer_embedding: 2D tensor of shape 1 x entity_emb_dim
        :param features_mask: 1D tensor of length num_features
        :return: 2D tensor of shape 1 x 'no. of candidate_entities'
        """
        
        bert_question_embedding = self.W_BERT(bert_question_embedding)
        bert_question_embedding = bert_question_embedding.unsqueeze(dim = 0)
        question_embedding = torch.cat((bert_question_embedding, kewer_question_embedding), 1)
        question_embedding = self.W_CAT(question_embedding)
        
        # question_embedding = kewer_question_embedding + bert_question_embedding 
        # question_embedding = question_embedding / torch.linalg.norm(question_embedding)
        
        if self.interaction == 'mult':
            p_interaction = self.mult_interaction(question_embedding, self.W_p,
                                                  p_inputs)  # shape: 1 x no. of candidate entities x 1
            lit_interaction = self.mult_interaction(question_embedding, self.W_lit,
                                                    lit_inputs)  # shape: 1 x no. of candidate entities x 1
            cat_interaction = self.mult_interaction(question_embedding, self.W_cat,
                                                    cat_inputs)  # shape: 1 x no. of candidate entities x 1
            ent_interaction = self.mult_interaction(question_embedding, self.W_ent,
                                                    ent_inputs)  # shape: 1 x no. of candidate entities x 1
            s_interaction = self.mult_interaction(previous_answer_embedding, self.W_s,
                                                  s_inputs)  # shape: 1 x no. of candidate entities x 1
        elif self.interaction == 'add':
            p_interaction = self.add_interaction(question_embedding, p_inputs, self.Wq_p, self.Ws_p, self.v_p)
            lit_interaction = self.add_interaction(question_embedding, lit_inputs, self.Wq_lit, self.Ws_lit, self.v_lit)
            cat_interaction = self.add_interaction(question_embedding, cat_inputs, self.Wq_cat, self.Ws_cat, self.v_cat)
            ent_interaction = self.add_interaction(question_embedding, ent_inputs, self.Wq_ent, self.Ws_ent, self.v_ent)
            s_interaction = self.add_interaction(previous_answer_embedding, s_inputs, self.Wq_s, self.Ws_s, self.v_s)
        elif self.interaction == 'dot':
            p_interaction = self.dot_interaction(question_embedding,
                                                 p_inputs)  # shape: 1 x no. of candidate entities x 1
            lit_interaction = self.dot_interaction(question_embedding,
                                                   lit_inputs)  # shape: 1 x no. of candidate entities x 1
            cat_interaction = self.dot_interaction(question_embedding,
                                                   cat_inputs)  # shape: 1 x no. of candidate entities x 1
            ent_interaction = self.dot_interaction(question_embedding,
                                                   ent_inputs)  # shape: 1 x no. of candidate entities x 1
            s_interaction = self.dot_interaction(previous_answer_embedding,
                                                 s_inputs)  # shape: 1 x no. of candidate entities x 1
        entity_features = torch.cat((overlap_features, p_interaction, lit_interaction, cat_interaction, ent_interaction,
                                     s_interaction), -1)  # shape: 1 x no. of candidate entities x 10
        # assert entity_features.shape == (1, overlap_features.shape[1], 10)
        entity_scores = self.model_cosine(entity_features, features_mask)
        # assert entity_scores.shape == (1, overlap_features.shape[1])
        return entity_scores