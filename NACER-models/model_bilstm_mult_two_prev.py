import torch
import torch.nn as nn

from model_mult_two_prev import ModelMult
from infersent import InferSent
from utils import all_questions

infersent_model_path = 'data/encoder/infersent1.pkl'
glove_path = 'data/GloVe/glove.840B.300d.txt'


class ModelBiLSTMMult(nn.Module):
    """The model that combines Bi-LSTM encoder of question and ModelMult"""

    def __init__(self, interaction: str, same_w: bool = True, entity_emb_dim: int = 300,
                 interaction_dim: int = 512, num_hidden: list = None):
        """Initialize the InferSent and ModelMult objects."""
        super(ModelBiLSTMMult, self).__init__()
        self.infersent = InferSent({'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                                    'pool_type': 'max', 'dpout_model': 0.0, 'version': 1})
        self.infersent.load_state_dict(torch.load(infersent_model_path))
        self.infersent.set_w2v_path(glove_path)
        self.infersent.build_vocab(all_questions())
        self.model_mult = ModelMult(interaction, same_w=same_w, question_emb_dim=4096, entity_emb_dim=entity_emb_dim,
                                    interaction_dim=interaction_dim, num_hidden=num_hidden)

    def forward(self, overlap_features: torch.Tensor, p_inputs: torch.Tensor, lit_inputs: torch.Tensor,
                cat_inputs: torch.Tensor, ent_inputs: torch.Tensor, s_inputs: torch.Tensor,
                question: str, previous_answer_embedding_first: torch.Tensor,previous_answer_embedding_second: torch.Tensor,previous_answer_embedding_avg: torch.Tensor,
                features_mask: torch.Tensor = None) -> torch.Tensor:
        """Calculate logit scores for each candidate entity.

        :param question: question text
        """
        question_embedding = self.infersent.encode_one(question)
        return self.model_mult(overlap_features, p_inputs, lit_inputs, cat_inputs, ent_inputs, s_inputs,
                               question_embedding, previous_answer_embedding_first,previous_answer_embedding_second,previous_answer_embedding_avg, features_mask)
