import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation Layer
    Applies conditional modulation using gamma and beta parameters
    """
    def __init__(self, in_dim: int, out_dim: int, cond_dim: int):
        super().__init__()
        # Base linear transformation
        self.linear = nn.Linear(in_dim, out_dim)
        # Networks to generate gamma and beta
        self.gamma_net = nn.Linear(cond_dim, out_dim)
        self.beta_net = nn.Linear(cond_dim, out_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim], c: [B, cond_dim]
        h = self.linear(x)  # [B, out_dim]
        gamma = self.gamma_net(c)  # [B, out_dim]
        beta = self.beta_net(c)    # [B, out_dim]
        return gamma * h + beta    # FiLM modulation


class CausalEncoder(nn.Module):
    """
    Variational Causal Encoder
    Infers latent variables U and environment variables E from data X
    """
    def __init__(self, input_dim: int, latent_dim: int, num_env: int):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 128), 
            nn.ReLU()
        )
        self.mu_u = nn.Linear(128, latent_dim)
        self.logvar_u = nn.Linear(128, latent_dim)
        self.logits_e = nn.Linear(128, num_env)
        
    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.shared_net(X)
        mu_U = self.mu_u(h)
        logvar_U = self.logvar_u(h)
        logits_E = self.logits_e(h)
        
        # Sample U from Gaussian distribution
        U = mu_U + torch.exp(0.5 * logvar_U) * torch.randn_like(mu_U)
        
        # Sample E from Gumbel-Softmax
        if self.training:
            E = F.gumbel_softmax(logits_E, tau=0.2, hard=False)
        else:
            E = F.gumbel_softmax(logits_E, tau=0.2, hard=True)
        
        return U, E, mu_U, logvar_U, logits_E

