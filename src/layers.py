import torch
import torch.nn as nn
from .dbf_core import power_iteration

def my_pack(x):
    x = (x == 1).to(torch.uint8)
    out = torch.zeros((x.shape[0]//8), device=x.device, dtype=torch.uint8)
    for i in range(8):
        out += x[i::8] << (7 - i)
    return out

@torch.compile
def my_unpack(x):
    out = torch.zeros((x.shape[0], 8), device=x.device, dtype=torch.int8)
    for i in range(8):
        out[:,i] = (x >> (7 - i)) & 1
    return out.flatten() * 2 - 1

def svd_abs2(W):
    Sg = W.sign()
    Sg[Sg == 0] = 1
    u, s, v = power_iteration(W.abs(), num_iters=5)
    # apx = s * torch.ger(u, v) # Original svd_abs returned this
    return u * s, Sg, v

class BitLinear(nn.Module):
    def __init__(self, b):
        super().__init__()
        
        #u, b, v = svd_abs(w.float())
        b_packed = my_pack(b.flatten())
        self.shape = b.shape
        #print(b)
        #print(my_unpack(b_packed).reshape(b.shape))
        
        self.register_buffer("bp", b_packed)

    
    def forward(self, x):
        return x.matmul(my_unpack(self.bp).reshape(self.shape).T.to(x.dtype))
        

        
class Mul(nn.Module):
    def __init__(self, w):
        super().__init__()
        #print("w", w.amin().item(), w.median().item(), w.amax().item())
        
        self.register_buffer("w", w)
    
    def forward(self, x):
        return x * self.w.to(x.dtype)
        

def replace(lx):
    dev = "cuda"
    m1 = lx.weight.B
    m2 = lx.weight.A
    
    u1, b1, v1 = svd_abs2(m1.float())
    u2, b2, v2 = svd_abs2(m2.float())
    
    lx2 = nn.Sequential(
        Mul(v1),
        BitLinear(b1),
        Mul(u1*lx.weight.mid*v2),
        BitLinear(b2),
        Mul(u2)
    )
    return lx2
