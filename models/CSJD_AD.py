import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional

from .modules import CausalEncoder, FiLMLayer


class CausalVariationalCSJDAD(nn.Module):
    """
    Causal Variational Neural Jump Diffusion Anomaly Detector
    Implements dual-path generation with explicit causal contrast loss
    """
    def __init__(self, 
                 input_dim: int, 
                 latent_dim: int = 64, 
                 num_env: int = 4,
                 hidden_dim: int = 128,
                 lambda_causal: float = 0.5,
                 lambda_kl: float = 0.05,
                 dt: float = 0.01,
                 gamma: float = 0.5,
                 features_dim: int = 128):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_env = num_env
        self.lambda_causal = lambda_causal  
        self.lambda_kl = lambda_kl
        self.dt = dt
        self.gamma = gamma
        self.features_dim = features_dim
        self.num_divide = 10
        
        # Components
        self.encoder = CausalEncoder(input_dim, latent_dim, num_env)
        self.mu_layer = FiLMLayer(latent_dim, latent_dim, num_env)
        self.sigma_layer = FiLMLayer(latent_dim, latent_dim, num_env)
        self.J_layer = FiLMLayer(latent_dim, latent_dim, num_env)
        # FiLM layer for jump probability followed by sigmoid
        self.p_layer = FiLMLayer(latent_dim, latent_dim, num_env)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, X: torch.Tensor, return_features: bool = False) -> dict:
        batch_size = X.size(0)
        # Step 1: Encode to latent variables
        U, E, mu_U, logvar_U, logits_E = self.encoder(X)
        

        # Counterfactual path (no jump)
        U_CF_trajectory, mu_E, sigma_E = self._euler_solver(U, E)
        
        # Extract final state from trajectory for path fusion
        U_CF = U_CF_trajectory[:, :, :, -1]  # [batch_size, seq_len, latent_dim]
        
        # Factual path (with jump)
        U_F, J_E, p_E = self._apply_jump_effect(U_CF_trajectory, E)
        
        # Path fusion
        U_final = U_CF + self.gamma * (U_F - U_CF)
        
        # Step 4: Decode to reconstruct data
        X_gen = self.decoder(U_final)
        

        
        return {
            'X_gen': X_gen,
            'U': U,
            'E': E,
            'U_CF': U_CF,
            'U_F': U_F,
            'U_final': U_final,
            'mu_U': mu_U,
            'logvar_U': logvar_U,
            'logits_E': logits_E,
            'mu_E': mu_E,
            'sigma_E': sigma_E,
            'J_E': J_E,
            'p_E': p_E,
        }
    

    def _euler_solver(self, U, E):
        """
        Euler-Maruyama solver for the continuous diffusion SDE:
        dU_t = μ(U_t)dt + σ(U_t)dW_t
        
        Note: Jump effects are applied separately outside this solver
        """
        # Get original shape for reshaping later
        original_shape = U.shape  # [batch_size, seq_len, latent_dim]
        batch_size, seq_len, latent_dim = original_shape
        
        # Reshape to process all timesteps at once
        U_reshaped = U.view(batch_size * seq_len, latent_dim)  # [batch_size * seq_len, latent_dim]
        E_reshaped = E.view(batch_size * seq_len, -1)  # [batch_size * seq_len, num_env]
        
        # Calculate dt for each sample in the batch
        dt = self.dt / self.num_divide
        
        # Initialize trajectory tensor
        eta_ts = torch.zeros(batch_size * seq_len, self.latent_dim, self.num_divide + 1, device=U.device)
        eta_ts[:, :, 0] = U_reshaped
        
        current_u = U_reshaped.clone()

        for step in range(self.num_divide):
            # Calculate drift μ(U_t) - continuous component
            drift = self.mu_layer(current_u, E_reshaped)
            
            # Calculate volatility σ(U_t) - continuous component
            volatility = self.sigma_layer(current_u, E_reshaped)
            
            # Generate Brownian motion increment: dW_t ~ N(0, dt)
            dW = torch.randn_like(current_u) * math.sqrt(dt)
            
            # Apply Euler-Maruyama update for continuous diffusion only:
            # U_{t+dt} = U_t + μ(U_t)dt + σ(U_t)dW_t
            u_next = current_u + drift * dt + volatility * dW
            
            # Store in trajectory
            eta_ts[:, :, step + 1] = u_next
            current_u = u_next

        # Reshape outputs back to original sequence format
        eta_ts_reshaped = eta_ts.view(batch_size, seq_len, self.latent_dim, self.num_divide + 1)
        drift_reshaped = drift.view(batch_size, seq_len, self.latent_dim)
        volatility_reshaped = volatility.view(batch_size, seq_len, self.latent_dim)
        return eta_ts_reshaped, drift_reshaped, volatility_reshaped
    

    def _apply_jump_effect(self, U, E):
        """
        Apply jump effect to the given state
        Returns: (u_with_jump, jump_magnitude, jump_probability, jump_occurred)
        """
        # Handle different input shapes
        if U.dim() == 4:  # From euler_solver: [batch_size, seq_len, latent_dim, num_divide + 1]
            # Take the final state from the trajectory
            U_final = U[:, :, :, -1]  # [batch_size, seq_len, latent_dim]
            original_shape = U_final.shape
            batch_size, seq_len, latent_dim = original_shape
            
            # Reshape to process all timesteps at once
            U_reshaped = U_final.view(batch_size * seq_len, latent_dim)
            E_reshaped = E.view(batch_size * seq_len, -1)
        else:  # Regular case: [batch_size, seq_len, latent_dim]
            original_shape = U.shape
            batch_size, seq_len, latent_dim = original_shape
            
            # Reshape to process all timesteps at once
            U_reshaped = U.view(batch_size * seq_len, latent_dim)
            E_reshaped = E.view(batch_size * seq_len, -1)
        
        # Calculate jump magnitude J(U_t)
        J_E = self.J_layer(U_reshaped, E_reshaped)
        
        # Calculate jump probability p(U_t) - simplified approach
        p_logits = self.p_layer(U_reshaped, E_reshaped)
        p_E = torch.sigmoid(p_logits)  # Simple sigmoid instead of Gumbel-Sigmoid
                
        # Apply jump: U_jump = U + J(U) * Bernoulli(p(U))
        U_F = U_reshaped + J_E * p_E
        
        # Reshape outputs back to original sequence format
        U_F_reshaped = U_F.view(batch_size, seq_len, latent_dim)
        J_E_reshaped = J_E.view(batch_size, seq_len, latent_dim)
        p_E_reshaped = p_E.view(batch_size, seq_len, latent_dim)
        
        return U_F_reshaped, J_E_reshaped, p_E_reshaped
    

    def compute_loss(self, X: torch.Tensor, outputs: dict) -> dict:
        """
        Compute total loss with explicit causal contrast
        """
        X_gen = outputs['X_gen']
        U_F = outputs['U_F']
        U_CF = outputs['U_CF']
        mu_U = outputs['mu_U']
        logvar_U = outputs['logvar_U']
        E = outputs['E']
        J_E = outputs['J_E']
        p_E = outputs['p_E']
        
        # Reconstruction loss
        L_recon = F.mse_loss(X_gen, X)
        
        # Fixed causal contrast loss
        # Encourage meaningful jump patterns while maintaining stability
        jump_magnitude = J_E.abs().mean()  # Scalar: average jump magnitude
        path_divergence = ((U_F - U_CF) ** 2).mean()  # Scalar: average path divergence
        
        # Causal loss: encourage correlation between jumps and path changes
        # Use positive formulation with proper scaling
        L_causal = F.mse_loss(U_F, U_CF)  # Basic path divergence loss
        
        # For monitoring - compute anomaly score for consistency
        anomaly_score = J_E.abs().sum(-1) * (1-p_E).mean(-1)
        L_causal = L_causal * anomaly_score.mean()
        
        # KL divergence losses (Variational)
        L_KL_U = -0.5 * torch.mean(1 + logvar_U - mu_U.pow(2) - logvar_U.exp())
        L_KL_E = (E * torch.log(E * self.num_env + 1e-8)).mean()  # Entropy regularizer for E
        

        # Total loss with proper scaling and clamping
        L_total = L_recon + self.lambda_causal * L_causal + self.lambda_kl * (L_KL_U + L_KL_E) 
        
        # Debug prints
        if torch.isnan(L_total) or torch.isinf(L_total):
            print(f"DEBUG: L_total is NaN/Inf!")
            print(f"L_recon: {L_recon.item()}, L_causal: {L_causal.item()}, L_KL_U: {L_KL_U.item()}, L_KL_E: {L_KL_E.item()}")
            print(f"J_E stats: min={J_E.min().item()}, max={J_E.max().item()}, mean={J_E.mean().item()}")
            print(f"p_E stats: min={p_E.min().item()}, max={p_E.max().item()}, mean={p_E.mean().item()}")
        
        # Clamp total loss to prevent explosion
        L_total = torch.clamp(L_total, -100, 100)
        
       
        return {
            'total_loss': L_total,
            'reconstruction_loss': L_recon,
            'causal_loss': L_causal,
            'kl_u_loss': L_KL_U,
            'kl_e_loss': L_KL_E,
            'anomaly_score': anomaly_score,
            'consistency_loss': L_causal,
            'jump_magnitude': jump_magnitude,
            'path_divergence': path_divergence
        }
    
    def detect_anomalies(self, X: torch.Tensor) -> torch.Tensor:
        """
        Detect anomalies using reconstruction error enhanced with jump-diffusion dynamics
        Returns point-wise anomaly scores
        """
        with torch.no_grad():
            outputs = self.forward(X)
            
            recon_error = F.mse_loss(outputs['X_gen'], X, reduction='none').mean(-1)
            
            mu_U = outputs['mu_U']
            logvar_U = outputs['logvar_U']
            
            J_E = outputs['J_E']
            p_E = outputs['p_E']
            U_F = outputs['U_F']
            U_CF = outputs['U_CF']
            E = outputs['E']
            
            jump_score = J_E.abs().sum(-1) * (1-p_E).mean(-1)

            causal_score = F.mse_loss(U_F, U_CF)  # Basic path divergence loss
            jump_anomaly_score = J_E.abs().sum(-1) * (1-p_E).mean(-1)
            causal_score *= jump_anomaly_score.mean()
            L_KL_U = -0.5 * torch.mean(1 + logvar_U - mu_U.pow(2) - logvar_U.exp())            
            L_KL_E = (E * torch.log(E * self.num_env + 1e-8)).mean()  # Entropy regularizer for E
            
        
            anomaly_score = recon_error + self.lambda_causal * causal_score + self.lambda_kl * (L_KL_U + L_KL_E) 
            return anomaly_score
    
    def predict_score(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict anomaly scores (alias for detect_anomalies for compatibility)
        """
        return self.detect_anomalies(X)


def create_causal_variational_csjdad(input_dim: int, 
                                   latent_dim: int = 64,
                                   num_env: int = 4,
                                   hidden_dim: int = 128,
                                   lambda_causal: float = 0.5,
                                   lambda_kl: float = 0.05,
                                   dt: float = 0.01,
                                   gamma: float = 0.5) -> CausalVariationalCSJDAD:
    """
    Factory function to create CausalVariationalCSJDAD model
    """
    return CausalVariationalCSJDAD(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_env=num_env,
        hidden_dim=hidden_dim,
        lambda_causal=lambda_causal,
        lambda_kl=lambda_kl,
        dt=dt,
        gamma=gamma
    ) 