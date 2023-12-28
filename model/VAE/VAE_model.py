import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

import sys
sys.path.append('./')
from model.diffusion_model import Block



class Encoder(nn.Module):
    def __init__(self,
                 input_dim=2000,
                 hidden_dim=1024,
                 mu_var_dim=10
                 ) -> None:
        super().__init__()
        
        self.enc = nn.Sequential(
             Block(input_dim=input_dim, 
                  output_dim=hidden_dim,
                  norm='bn',
                  act='relu'
                  )
        )
        
        self.mu_enc = nn.Sequential(
            Block(input_dim=hidden_dim, 
                  output_dim=mu_var_dim,
                  norm='',
                  )
        )
        self.var_enc = nn.Sequential(
            Block(input_dim=hidden_dim, 
                  output_dim=mu_var_dim,
                  norm='',
                  )
        )
    def reparameterize(self, mu, var):
        return Normal(mu, var.sqrt()).rsample()
    
    def forward(self, x, y=None):
        h = self.enc(x)
        mu = self.mu_enc(h)
        var = torch.exp(self.var_enc(h))
        z = self.reparameterize(mu, var)
        return z, mu, var
    
    
class VAE(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 mu_var_dim=10) -> None:
        super().__init__()
        
        self.encoder = Encoder(input_dim=input_dim,
                               hidden_dim=hidden_dim,
                               mu_var_dim=mu_var_dim)
        
        self.decoer = Block(
            input_dim=mu_var_dim,
            output_dim=input_dim,
            norm='dsbn',
            act='sigmoid'
        )
        
        
        