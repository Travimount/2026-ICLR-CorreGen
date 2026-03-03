import torch

def init_matrix(C1, C2, p, q, loss_fun="square_loss"):
    fC1, fC2, hC1, hC2 = _transform_matrix(C1, C2, loss_fun)
    constC1 = fC1 @ p.reshape(-1, 1)
    constC2 = q.reshape(1, -1) @ fC2.T
    constC = constC1 + constC2

    return constC, hC1, hC2

def _transform_matrix(C1,C2,loss_fun="square_loss"):
    if loss_fun == "square_loss":
        def f1(a):
            return a**2

        def f2(b):
            return b**2

        def h1(a):
            return a

        def h2(b):
            return 2 * b
    elif loss_fun == "kl_loss":
        def f1(a):
            return a * torch.log(a + 1e-18) - a

        def f2(b):
            return b

        def h1(a):
            return a

        def h2(b):
            return torch.log(b + 1e-18)
    else:
        raise ValueError(
            f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}."
        )

    fC1 = f1(C1)
    fC2 = f2(C2)
    hC1 = h1(C1)
    hC2 = h2(C2)
    return fC1, fC2, hC1, hC2

def sinkhorn(p, q, C, reg, niter=50,thresh=1e-6, log=False, warmstart=None,gamma=None,semi_use=False,is_partial=False,mask=False):
    if log:
        log = {"err": []}

    def M(u, v):
        M_mat = (-C + u.unsqueeze(1) + v.unsqueeze(0)) / reg
        if mask:
            M_mat.fill_diagonal_(-1e6)
        return M_mat

    if warmstart is not None:
        u, v = warmstart
        if not isinstance(u, torch.Tensor) or not isinstance(v, torch.Tensor):
            raise ValueError("Warmstart must be a tuple of tensors (u, v).")
    else:
        u = torch.zeros_like(p,device=p.device)
        v = torch.zeros_like(q,device=q.device)

    err = 0.
    actual_nits = 0

    for i in range(niter):
        u1 = u.clone()
        u = reg * (torch.log(p) - torch.logsumexp(M(u, v), dim=1, keepdim=True).squeeze()) + u
        v_temp = reg * (torch.log(q) - torch.logsumexp(M(u, v).mT, dim=1, keepdim=True).squeeze())
        if not semi_use:
            v = v_temp + v
        else:
            a = gamma / (reg + gamma)
            if is_partial:
                v[:-1] = a * (v_temp[:-1] + v[:-1])
                v[-1] = v_temp[-1] + v[-1]
            else:
                v = a * (v_temp + v)
        err = torch.norm(u - u1)
        actual_nits += 1
        if err < thresh:
            break
        # if i == niter - 1 and err > thresh * 10000:
        #     print(f"Warning: Sinkhorn algorithm did not converge after {niter} iterations. Error: {err.item()}")
    if log:
        log["niter"] = actual_nits
        log["u"] = u
        log["v"] = v
    pi = torch.exp(M(u, v)).float()

    # has_nan = torch.isnan(pi).any() and torch.isinf(pi).any()
    # if has_nan:
    #     raise ValueError("Sinkhorn algorithm produced NaN values in the transport plan.")

    if log:
        return pi, log
    else:
        return pi

def tensor_product(constC, hC1, hC2, T):
    A = - (hC1 @ T @ hC2.T)
    tens = constC + A
    return tens

def gwggard(constC, hC1, hC2, T):
    return 2 * tensor_product(constC, hC1, hC2, T)

def gwloss(constC, hC1, hC2, T):
    tens = tensor_product(constC, hC1, hC2, T)
    return torch.sum(T * tens)