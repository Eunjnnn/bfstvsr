import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.modules.SIREN import SineLayer, Siren

import cupy
import collections

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


# B-spline 3rd order kernel
kernel_code = """
extern "C" __global__
void bspline_3rd_kernel(const int B, const int qs, const int C, const float *x, float *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < B * qs * C) {
        const int b = idx / (qs * C);
        const int q = (idx % (qs * C)) / C;
        const int c = idx % C;
        
        float val = x[idx];
        float result = 0.0;

        // Compute the 3rd order B-spline value based on the ranges
        if (val <= -2.0) {
            result = 0.0;
        } else if (val <= -1.0) {
            result = ((2.0 + val) * (2.0 + val) * (2.0 + val)) / 6.0;
        } else if (val <= 0.0) {
            result = (4.0 - 6.0 * val * val - 3.0 * val * val * val) / 6.0;
        } else if (val <= 1.0) {
            result = (4.0 - 6.0 * val * val + 3.0 * val * val * val) / 6.0;
        } else if (val <= 2.0) {
            result = ((2.0 - val) * (2.0 - val) * (2.0 - val)) / 6.0;
        } else {
            result = 0.0;
        }

        out[idx] = result;
    }
}
"""

@cupy.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    # return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction) 
    return cupy.RawModule(code=strKernel).get_function(strFunction)
# end

def bspline_3rd_cupy(x):
    B, qs, C = x.shape
    
    x = x.contiguous(); assert(x.is_cuda == True)
    output = x.new_zeros([B, qs, C])
    
    if x.is_cuda == True:
        n = output.nelement()
        kernel = 'bspline_3rd_kernel' 
        cupy_launch(kernel, kernel_code)(
            grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
            block=tuple([ 512, 1, 1 ]),
            args=[ cupy.int32(B), cupy.int32(qs), cupy.int32(C), x.data_ptr(), output.data_ptr() ],
            stream=collections.namedtuple('Stream', 'ptr')(torch.cuda.current_stream().cuda_stream)
        )
    # Reshape the output to the original shape (B, qs, C)
    return output

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
        self.q_coef = None
        
        
    def forward(self, feats, ts, init=False, t_res=None):
        
        B, N = ts.shape
        _, qs, _ = feats.shape
        
        t_res = self.t_res if t_res is None else t_res
        
        tf = ts.reshape(B*N, 1, 1).repeat(1, qs, self.bspline_features) # (B*N, qs, bspline_features)
        tb = 1-tf
        t = torch.concat([tf, tb]) # (2*B*N, qs, bspline_features)
        
        if init:
            feats = self.net(feats).repeat(1, N, 1).reshape(2*B*N, qs, -1)
            self.dillation = self.t_dilation(t_res).reshape(1, 1, -1).repeat(2*B*N, 1, 1)
            self.q_coef, self.q_knot = torch.split(feats, self.bspline_features, dim=-1)
        
        q_knot = t-self.q_knot        
        q_knot = self.dillation * q_knot
        # q_knot = spline_basis(q_knot, self.basis_ord)
        
        q_knot = bspline_3rd_cupy(q_knot)
        
        inp = self.q_coef * q_knot
        output = self.final_linear(inp)
                
        return output