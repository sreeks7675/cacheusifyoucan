import torch
import torch.nn as nn


class ConsensusModel(nn.Module):

    def __init__(self, feature_dim=3846):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 2)

        )

    def forward(self, features):

        return self.network(features)
