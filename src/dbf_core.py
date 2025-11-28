import torch
import time
import numpy as np

def power_iteration(A, num_iters=5):
    """
    Performs power iteration to compute the top singular vectors and value.
    
    Arguments:
        A (torch.Tensor): The input matrix of shape (m, n).
        num_iters (int): Number of iterations to perform.
    
    Returns:
        u (torch.Tensor): Dominant left singular vector (m,).
        sigma (torch.Tensor): Dominant singular value (scalar).
        v (torch.Tensor): Dominant right singular vector (n,).
    """
    # Start with a random vector on the appropriate device
    n = A.shape[1]
    v = torch.randn(n, device=A.device)
    v = v / torch.norm(v)
    
    for _ in range(num_iters):
        # Multiply A*v
        u = torch.mv(A, v)
        u_norm = torch.norm(u)
        if u_norm == 0:
            break
        u = u / u_norm
        
        # Multiply A^T*u
        v = torch.mv(A.t(), u)
        v_norm = torch.norm(v)
        if v_norm == 0:
            break
        v = v / v_norm
    
    # Estimate the dominant singular value as ||A*v||
    sigma = torch.norm(torch.mv(A, v))
    # The left singular vector corresponding to sigma:
    u = torch.mv(A, v) / sigma
    return u, sigma, v

def svd_abs(W):
    Sg = W.sign()
    Sg[Sg == 0] = 1
    u, s, v = power_iteration(W.abs(), num_iters=5)
    apx = s * torch.ger(u, v)
    
    return apx * Sg

def find_other2(A, W, nnz, Z, U, print_sc=None, debug=False, reg=0, rho_start=0.03, iters=3, prune_iters=1, flip=False, final=False):
    XX = A.T.matmul(A)
    XX += torch.diag(torch.ones_like(XX.diag())) * XX.diag().mean() * reg
    
    #norm2 = torch.ones_like(norm2)
    Wnn = W# * norm2.unsqueeze(1)
    rho = 1
    XY = A.T.matmul(Wnn)
    XXinv = torch.inverse(XX + torch.eye(XX.shape[1], device=XX.device)*rho)
    XXinv2 = torch.inverse(XX + torch.eye(XX.shape[1], device=XX.device)*rho_start)
    U = U
    Z = Z
    
    B = XXinv2.matmul(XY + rho_start*(Z-U))
    
    r_scale = c_scale = mask = None

    for itt in range(iters-1):
        Z = svd_abs(B+U)
        #print("   ", "z", itt, (A.matmul(Z) - W).square().sum().item(), W.square().sum().item())

        U = U + (B - Z)    

        B = XXinv.matmul(XY + rho*(Z-U))

    Z = svd_abs(B+U)
    #print("   ", "z", iters-1, (A.matmul(Z) - W).square().sum().item(), W.square().sum().item())
    U = U + (B - Z)
   
    return (Z), U


def factorizeT(W, XX, o_norm, asp=0.16, sp=0.16, iters=80):
    nza = int(W.shape[0]**2 * asp)
    nzb = int(W.numel() * sp - nza)
    
    norm = XX.sqrt().unsqueeze(1) + 1e-8
    norm_o = o_norm.sqrt() + 1e-8
       
    Wn = W * norm * norm_o
       
    mid = int(1.5*(W.shape[0]*W.shape[1]) / (W.shape[0] + W.shape[1]))
    
    Az = torch.randn((W.shape[0], mid), device=W.device)
    Au = torch.zeros_like(Az)

    Bz = torch.randn((mid, W.shape[1]), device=W.device)
    Bu = torch.zeros_like(Bz)
    
    for itt in range(iters):
        rho_start = min(1.0, itt / (iters-3))**3
        if True or itt > iters // 2:
            nzaa = nza
            nzbb = nzb
        else:
            alph = (itt / (iters // 2))**2
            nzaa = int(nza / 2 * (1-alph) + nza * alph)
            nzbb = int(nzb / 2 * (1-alph) + nzb * alph)

            
        mid = Bz.norm(dim=1) + 1e-12
        final = itt == iters - 1
        
        Az, Au = (x.T for x in find_other2(Bz.T / mid, Wn.T, nzaa, Az.T, Au.T, reg=3e-2, debug=False, rho_start=rho_start, flip=True, final=final))
        mid = Az.norm(dim=0) + 1e-12
        Bz, Bu = find_other2(Az / mid, Wn, nzbb, Bz, Bu, reg=3e-2, debug=False, rho_start=rho_start, final=final)
        #print("err", itt, ((Az / mid).matmul(Bz) - Wn).square().sum().item(), (Wn).square().sum().item())
        if itt == iters - 1:
            print("err", itt, ((Az / mid).matmul(Bz) - Wn).square().sum().item(), (Wn).square().sum().item())
            
    return ((Az / norm).matmul(Bz / norm_o)).T, (Bz / norm_o).T, (Az / norm).T, 1 / mid


def factorizef(W, XX, o_norm, asp=0.16, sp=0.16, iters=80, l_prev=None, target_mid=None):
    s_time = time.time()
    if W.shape[0] >= W.shape[1]:
        return factorizeT(W.T, XX, o_norm, asp, sp=sp, iters=iters)
    
    #print("a")
    nza = int(W.shape[0]**2 * asp)
    nzb = int(W.numel() * sp - nza)
    norm = XX.sqrt() + 1e-8
    norm_o = (o_norm.sqrt() + 1e-8).unsqueeze(1)

    Wn = W * norm * norm_o
    
    if target_mid is not None:
        mid = target_mid
    else:
        mid = int(1.5*(W.shape[0]*W.shape[1]) / (W.shape[0] + W.shape[1]))
    
    Az = torch.randn((W.shape[0], mid), device=W.device)
    Au = torch.zeros_like(Az)

    Bz = torch.randn((mid, W.shape[1]), device=W.device)
    Bu = torch.zeros_like(Bz)
    
    for itt in range(iters):
        rho_start = min(1.0, itt / (iters-3))**3
        if True or itt > iters // 2:
            nzaa = nza
            nzbb = nzb
        else:
            alph = (itt / (iters // 2))**2
            nzaa = int(nza / 2 * (1-alph) + nza * alph)
            nzbb = int(nzb / 2 * (1-alph) + nzb * alph)
            
        
        final = itt == iters - 1
        mid = Bz.norm(dim=1) + 1e-12
        Az, Au = (x.T for x in find_other2(Bz.T / mid, Wn.T, nzaa, Az.T, Au.T, reg=3e-2, debug=False, rho_start=rho_start, final=final))        
        mid = Az.norm(dim=0) + 1e-12
        Bz, Bu = find_other2(Az / mid, Wn, nzbb, Bz, Bu, reg=3e-2, debug=False, rho_start=rho_start, flip=True, final=final)
        
        if itt == iters - 1:
            print("err", itt, ((Az / mid).matmul(Bz) - Wn).square().sum().item(), (Wn).square().sum().item())
            
    return (Az / norm_o).matmul(Bz / norm), Az / norm_o, Bz / norm, 1 / mid

def factorize(lx, l_prev=None, target_mid=None):
    W = lx.weight.detach().float()
    
    sm = min(W.shape)
    lg = max(W.shape)
    mid = sm
    print("mid", mid)
    lim = 1.25
    for density in np.linspace(0.1, 0.4, 301):
        total_size = 0
        total_pars = 0
        total_pars += sm*lg
        mask_size = sm*mid + mid*lg
        total_ones2 = sm*lg * density
        p=total_ones2 / mask_size
        ent = -p*np.log2(p)-(1-p)*np.log2(1-p)
        total_size += (total_ones2*2 + mask_size*ent)
        #print(" ", density, total_size / total_pars)
        if total_size / total_pars < lim:
            sp = density
    
    
    if W.shape[0] == W.shape[1]:
        asp = sp/2
    else:
        asp = sp
    W2, Ab, Bb, mid = factorizef(W, lx.i_norm, lx.o_norm, asp=asp, sp=sp, l_prev=l_prev, iters=260, target_mid=target_mid)
    Ac = Ab
    
    #Ac = Ab
    #W3 = Ac.matmul(Bb)
    
    #Bc = get_at(lx.XX, W.T, Bb.T, Ac.T).T
    Bc = Bb
    
    An = Ac.norm() + 1e-12
    Bn = Bc.norm() + 1e-12
    Ac *= (Bn/An).sqrt()
    Bc *= (An/Bn).sqrt()
    
    W3 = (Ac * mid).matmul(Bc)
    assert W3.shape == lx.weight.shape
    print("sparsity check", ((Ac != 0).sum() + (Bb != 0).sum()).item() / W3.numel())
    return W3, Ac, Bc, mid
