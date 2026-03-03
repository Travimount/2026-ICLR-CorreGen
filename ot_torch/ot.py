import torch
import numpy as np
import math
import warnings
from ot_torch.utils import *

def entropic_wasserstein(M,
               p=None,
               q=None,
               reg=0.1,
               log=False,
               mask=False,
               ):

    if p is None:
        p = torch.full((M.shape[0],), 1 / M.shape[0], dtype=M.dtype, device=M.device)
    if q is None:
        q = torch.full((M.shape[1],), 1 / M.shape[1], dtype=M.dtype, device=M.device)

    N1, N2 = M.shape[0], M.shape[1]
    mu = torch.full((N1,), -math.log(N1), dtype=M.dtype, device=M.device)
    nu = torch.full((N2,), -math.log(N2), dtype=M.dtype, device=M.device)

    if log:
        log = {"err": []}

    T = sinkhorn(
        p,
        q,
        M,
        reg,
        warmstart=(mu, nu),
        mask=mask,
    )
    if abs(torch.sum(T) - 1) > 1e-5:
        warnings.warn(
            "Solver failed to produce a transport plan. You might "
            "want to increase the regularization parameter `epsilon`."
        )

    has_nan = torch.isnan(T).any() and torch.isinf(T).any()
    if has_nan:
        raise ValueError("Sinkhorn algorithm produced NaN values in the transport plan.")

    if log:
        log["wgw_dist"] = torch.sum(M * T)
        return T, log
    else:
        return T