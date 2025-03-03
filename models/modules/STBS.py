import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.modules.SIREN import SineLayer, Siren

class SplineBasis_2nd(torch.autograd.Function):    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        t = torch.where(x <= -1.5, 0.0 * x, 0.5 * (1.5 + x)**2)
        t = torch.where(x <= -0.5, t, 0.75 - x**2)
        t = torch.where(x <= 0.5, t, 0.5 * (1.5 - x)**2)
        w = torch.where(x > 1.5, 0.0 * x, t)
        return w
    
    @staticmethod
    def backward(ctx, grad_in):
        x, = ctx.saved_tensors
        t = torch.where(x <= -1.5, 0.0 * x, 1.5 + x)
        t = torch.where(x <= -0.5, t, -2 * x)
        t = torch.where(x <= 0.5, t, -(1.5 - x))
        w = torch.where(x > 1.5, 0.0 * x, t)
        return w * grad_in

class SplineBasis_3rd(torch.autograd.Function):    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        t = torch.where(x <= -2.0, 0.0 * x, ((2.0 + x)**3) / 6.0)
        t = torch.where(x <= -1.0, t, (4.0 - 6.0 * (x**2) - 3.0 * (x**3)) / 6.0)
        t = torch.where(x <= 0.0, t, (4.0 - 6.0 * (x**2) + 3.0 * (x**3)) / 6.0)
        t = torch.where(x <= 1.0, t, ((2.0 - x)**3) / 6.0)
        w = torch.where(x > 2.0, 0.0 * x, t)
        return w
    
    @staticmethod
    def backward(ctx, grad_in):
        x, = ctx.saved_tensors
        t = torch.where(x <= -2.0, 0.0 * x, 0.5 * ((2.0 + x)**2))
        t = torch.where(x <= -1.0, t, -2.0 * x - 1.5 * (x**2))
        t = torch.where(x <= 0.0, t, -2.0 * x + 1.5 * (x**2))
        t = torch.where(x <= 1.0, t, -0.5 * ((2.0 - x)**2))
        w = torch.where(x > 2.0, 0.0 * x, t)
        return w * grad_in

class SplineBasis_4th(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        t = torch.where(x <= -2.5, 0.0 * x, ((x**4) + 10 * (x**3) + 37.5 * (x**2) + 62.5 * x + 39.0625)/24.)
        t = torch.where(x <= -1.5, t, (-4 * (x**4) - 20 * (x**3) - 30 * (x**2) - 5 * x + 13.75)/24.)
        t = torch.where(x <= -0.5, t, (6 * (x**4) - 15 * (x**2) + 14.375)/24.)
        t = torch.where(x <= 0.5, t, (-4 * (x**4) + 20 * (x**3) - 30 * (x**2) + 5 * x + 13.75)/24.)
        t = torch.where(x <= 1.5, t, ((x**4) - 10 * (x**3) + 37.5 * (x**2) - 62.5 * x + 39.0625)/24.)
        w = torch.where(x > 2.5, 0.0 * x, t)
        return w
    
    @staticmethod
    def backward(ctx, grad_in):
        x, = ctx.saved_tensors
        t = torch.where(x <= -2.5, 0.0 * x, (4 * (x**3) + 30 * (x**2) + 75 * x + 62.5)/24.)
        t = torch.where(x <= -1.5, t, (-16 * (x**3) - 60 * (x**2) - 60 * x - 5)/24.)
        t = torch.where(x <= -0.5, t, (24 * (x**3) - 30 * x)/24.)
        t = torch.where(x <= 0.5, t, (-16 * (x**3) + 60 * (x**2) - 60 * x + 5)/24.)
        t = torch.where(x <= 1.5, t, (4 * (x**3) - 30 * (x**2) + 75 * x - 62.5)/24.)
        w = torch.where(x > 2.5, 0.0 * x, t)
        return w * grad_in
    
def spline_basis(x, basis_ord=3):
    if basis_ord==2:
        x.clamp_(-1.5 + 1e-6, 1.5 - 1e-6)
        return SplineBasis_2nd.apply(x)
    elif basis_ord==3:
        x.clamp_(-2.0 + 1e-6, 2.0 - 1e-6)
        return SplineBasis_3rd.apply(x)
    elif basis_ord==4:
        x.clamp_(-2.5 + 1e-6, 2.5 - 1e-6)
        return SplineBasis_4th.apply(x)
    else:
        raise ValueError

class STBSplineMapper(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, bspline_features, out_features, intermediate_linear=True, basis_ord=3, polar_coord=False, first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
    
        # B-Spline Features
        self.bspline_features = bspline_features
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features[0], 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features[i], hidden_features[i + 1], is_first=False, omega_0=hidden_omega_0))

        if intermediate_linear:
            intermediate_linear = nn.Linear(hidden_features[-1], 2*bspline_features)
            
            with torch.no_grad():
                intermediate_linear.weight.uniform_(-np.sqrt(6 / hidden_features[-1]) / hidden_omega_0, np.sqrt(6 / hidden_features[-1]) / hidden_omega_0)
                
            self.net.append(intermediate_linear)
        else:
            self.net.append(SineLayer(hidden_features[-1], 2*bspline_features, is_first=False, omega_0=hidden_omega_0))
        self.net = nn.Sequential(*self.net)
        
        self.t_dilation = nn.Linear(1, bspline_features, bias=False)
        self.final_linear = nn.Linear(bspline_features, out_features)
        
        self.basis_ord = basis_ord
        
        t_res = torch.tensor([0.125])
        self.register_buffer('t_res', t_res)
        
        self.polar_coord = polar_coord
        
    def forward(self, feats, ts, t_res=None):
        
        B, N = ts.shape
        _, qs, _ = feats.shape
        
        t_res = self.t_res if t_res is None else t_res
        
        tf = ts.reshape(B*N, 1, 1).repeat(1, qs, self.bspline_features) # (B*N, qs, bspline_features)
        tb = 1-tf
        t = torch.concat([tf, tb]) # (2*B*N, qs, bspline_features)
        
        feats = self.net(feats).repeat(1, N, 1).reshape(2*B*N, qs, -1)
        dillation = self.t_dilation(t_res).reshape(1, 1, -1).repeat(2*B*N, 1, 1)
        q_coef, q_knot = torch.split(feats, self.bspline_features, dim=-1)
        
        q_knot = t-q_knot        
        q_knot = dillation * q_knot
        q_knot = spline_basis(q_knot, self.basis_ord)
        
        inp = q_coef * q_knot
        output = self.final_linear(inp)
        
        # if self.polar_coord:
        #     output[..., 0] = F.relu(output[..., 0])
        
        return output