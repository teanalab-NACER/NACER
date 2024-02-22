import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelCosine(nn.Module):
    """The model for the method that takes feature values as input"""
    def __init__(self, num_features: int = 10, num_hidden: list = None):
        """Initialize the model object.

        num_features: number of input features
        num_hidden: list with sizes of each hidden layer (at least one)
        """
        super(ModelCosine, self).__init__()
        if num_hidden is None:
            num_hidden = [20, 10]
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_features, num_hidden[0]))
        for i in range(len(num_hidden) - 1):
            self.layers.append(nn.Linear(num_hidden[i], num_hidden[i + 1]))
        self.layers.append(nn.Linear(num_hidden[-1], 1))

    def forward(self, entity_features: torch.Tensor, features_mask: torch.Tensor = None) -> torch.Tensor:
        """Calculate logit scores for each candidate entity.

        entity_features: 3D tensor of shape 1 x 'no. of candidate entities' x num_features
        features_mask: 1D tensor of length num_features
        returns 2D tensor of shape 1 x 'no. of candidate_entities'
        """
        if features_mask is not None:
            entity_features = entity_features * features_mask
        hidden = self.layers[0](entity_features)
        hidden = F.relu(hidden)  # shape: 1 x no. of candidate entities x num_hidden[0]
        for layer in self.layers[1:-1]:
            hidden = layer(hidden)
            hidden = F.relu(hidden)
        entity_scores = self.layers[-1](hidden)  # shape: 1 x no. of candidate entities x 1
        entity_scores = torch.squeeze(entity_scores, dim=-1)  # shape: 1 x no. of candidate_entities
        return entity_scores
