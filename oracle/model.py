"""Neural network model definition."""

from typing import List
import torch
import torch.nn as nn


class OracleNet(nn.Module):
    """ORACLE neural network for solubility prediction."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256, 128], dropout: float = 0.0) -> None:
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(features)