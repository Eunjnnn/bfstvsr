import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.modules.SIREN import SineLayer, Siren


class STFSFourierMapper(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, fourier_features, out_features, intermediate_linear=True, first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
    
        # B-Spline Features
        self.fourier_features = fourier_features
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features[0], 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features[i], hidden_features[i + 1], is_first=False, omega_0=hidden_omega_0))

        if intermediate_linear:
            intermediate_linear = nn.Linear(hidden_features[-1], 2*fourier_features)
            
            with torch.no_grad():
                intermediate_linear.weight.uniform_(-np.sqrt(6 / hidden_features[-1]) / hidden_omega_0, np.sqrt(6 / hidden_features[-1]) / hidden_omega_0)
                
            self.net.append(intermediate_linear)
        else:
            self.net.append(SineLayer(hidden_features[-1], 2*fourier_features, is_first=False, omega_0=hidden_omega_0))
        self.net = nn.Sequential(*self.net)
        
        self.coef = nn.Conv2d(fourier_features, out_features, 3, padding=1)
        self.freq = nn.Conv2d(fourier_features, out_features, 3, padding=1)
        self.phase = nn.Linear(1, fourier_features, bias=False)
        self.final_linear = nn.Linear(fourier_features, out_features)
        
        
        
    def forward(self, feats, rel_c):        # rel_c : torch.Size([2*B, HH, WW, 2])
        
        _, qs, c = feats.shape
        BB, HH, WW, _ = rel_c.shape

        feats = self.net(feats).reshape(BB, HH, WW, -1)#.permute(0,3,1,2)       # torch.Size([2*B, HH, WW, 128])
        # dillation = self.phase(rel_c).reshape(1, 1, -1).repeat(2*B, 1, 1)
        q_coef, q_freq = torch.split(feats, self.fourier_features, dim=-1)

        q_coef = self.coef(q_coef.permute(0,3,1,2)).permute(0,2,3,1)     # torch.Size([2*B, HH, WW, fourier_features])
        q_freq = self.freq(q_freq.permute(0,3,1,2)).permute(0,2,3,1)     # torch.Size([2*B, HH, WW, fourier_features])

        q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)     # torch.Size([2*B, HH, WW, 2, 32])
        q_freq = torch.mul(q_freq, rel_c.unsqueeze(-1))                  # torch.Size([2*B, HH, WW, 2, 32])
        q_freq = torch.sum(q_freq, dim=-2)                               # torch.Size([2*B, HH, WW, 32])
        
        q_freq = torch.cat((torch.cos(np.pi*q_freq), torch.sin(np.pi*q_freq)), dim=-1)      # torch.Size([2*B, HH, WW, 64])
        inp = torch.mul(q_coef, q_freq).reshape(BB, HH*WW, -1)     # torch.Size([2*B, HH * WW, fourier_features])
        output = self.final_linear(inp)
        
        return output