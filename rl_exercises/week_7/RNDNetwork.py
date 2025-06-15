from collections import OrderedDict

import torch
import torch.nn as nn


class RNDNetwork(nn.Module):

    def __init__(self, input_dimension: int, output_dimension: int, hidden_dimension: int = 128, n_layers: int = 2) -> None:
        """
        Parameters
        ----------
        input_dimension : int
            Dimensionality of observation space.
        output_dimension : int
            dimension of vector corresponding to each state that is used to compute the RND reward.
        hidden_dimension : int
            Hidden layer size.
        n_layers : int
            Number of hidden layers.
        """
        super().__init__()
        layers = []
        current_dim = input_dimension
        for _ in range(n_layers):
            layers.append(nn.Linear(current_dim, hidden_dimension))
            layers.append(nn.ReLU())
            current_dim = hidden_dimension
        layers.append(nn.Linear(hidden_dimension, output_dimension))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of states, shape (batch, input_dimension).

        Returns
        -------
        torch.Tensor
            RND vectors, shape (batch, output_dimension).
        """
        return self.net(x)
