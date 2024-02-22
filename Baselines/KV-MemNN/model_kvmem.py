import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import pipeline

class ModelKvmem(nn.Module):
    """Key-Value Memory Network from the paper 'Key-Value Memory Networks for Directly Reading Documents'"""

    def __init__(self, qemb: str, num_hops: int = 3, input_dim: int = 300, question_emb_dim: int = 768):
        """Initialize the model object.

        num_hops: number of hops H.
        input_dim: dimensionality of key and value embedding d.
        """
        super(ModelKvmem, self).__init__()
        self.qemb = qemb
        self.num_hops = num_hops
        self.input_dim = input_dim
        self.question_emb_dim = question_emb_dim
        
        self.BERT = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    
        for param in self.BERT.parameters():
            param.requires_grad = False
            
        self.W = nn.Linear(self.question_emb_dim, self.input_dim, bias=False)
        self.W_CAT = nn.Linear(self.input_dim*2, self.input_dim, bias=False)
        
        self.R = nn.ModuleList()
        for i in range(num_hops):
            self.R.append(nn.Linear(input_dim, input_dim, bias=False))

    def forward(self, kewer_question_embedding: torch.Tensor, bert_question_embedding: torch.Tensor, 
                key_embeddings: torch.Tensor, value_embeddings: torch.Tensor, 
                candidate_embeddings: torch.Tensor) -> torch.Tensor:
        """Calculate logit scores for each candidate entity.

        question_embedding: 2D tensor of shape 1 x input_dim
        key_embeddings: 3D tensor of shape 1 x 'no. of triple memory slots' x input_dim
        value_embeddings: 3D tensor of shape 1 x 'no. of triple memory slots' x input_dim
        candidate_embeddings: 3D tensor of shape 1 x 'no. of candidate entities' x input_dim
        returns 2D tensor of shape 1 x 'no. of candidate_entities'
        """
        if self.qemb == 'kewer':
            bert_question_embedding = self.W(bert_question_embedding)
            # bert_question_embedding = bert_question_embedding.unsqueeze(dim = 0)
            question_embedding = torch.cat((bert_question_embedding, kewer_question_embedding), 0)
            q = self.W_CAT(question_embedding)
        else:
            q = self.W(question_embedding)
        
        for R_j in self.R:
            p_hi = F.softmax(torch.sum(q * key_embeddings, -1, keepdim=True),
                             dim=-2)  # shape: 1 x no. of triple memory slots x 1
            o = torch.sum(p_hi * value_embeddings, dim=-2)  # shape: 1 x input_dim
            q = F.normalize(R_j(q + o), dim=-1)  # shape: 1 x input_dim
        candidate_embeddings_norm = F.normalize(candidate_embeddings,
                                                dim=-1)  # shape: 1 x no. of candidate_entities x input_dim
        entity_scores = torch.sum(q * candidate_embeddings_norm, -1)  # shape: 1 x no. of candidate_entities
        return entity_scores