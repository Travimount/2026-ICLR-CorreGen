import torch
import math
import warnings
from ot_torch.utils import sinkhorn


def Partial_ot( M,
                p=None,
                q=None,
                reg=0.1,
                rho=0.2,
                log=False,
                mask=False,
                semi_use=False
                ):

    cpt = 0
    err = 1
    if p is None:
        if semi_use:
            p = torch.full((M.shape[0],), 1.0 / M.shape[0], dtype=M.dtype, device=M.device)
        else:
            p = torch.full((M.shape[0] + 1,), 1.0 / M.shape[0], dtype=M.dtype, device=M.device)
            p[-1] = rho
    else:
        total_mass = p.sum()
        if total_mass != 1.0: p /= total_mass
        if not semi_use: p = torch.cat((p, torch.tensor([rho], dtype=p.dtype, device=p.device)), dim=0)

    if q is None:
        if semi_use:
            q = torch.full((M.shape[1] + 1,), (1 - rho) / M.shape[0], dtype=M.dtype, device=M.device)
            q[-1] = rho
        else:
            q = torch.full((M.shape[1] + 1,), 1.0 / M.shape[0], dtype=M.dtype, device=M.device)
            q[-1] = rho
    else:
        total_mass = q.sum()
        if total_mass != 1.0: q = q / total_mass
        q = torch.cat((q * (1 - rho) if semi_use else q, torch.tensor([rho], dtype=q.dtype, device=q.device)), dim=0)


    A = torch.abs(M.max())
    xi = 1e2 * A

    if semi_use:
        M = torch.cat((M, torch.full((M.shape[0],1), 0, device=M.device)), dim=1)
    else:
        M = torch.cat((M, torch.full((M.shape[0],1), 0, device=M.device)), dim=1)
        M = torch.cat((M, torch.full((1, M.shape[1]), 0, device=M.device)), dim=0)
        M[-1, -1] = 2 * xi + A

    N1, N2 = M.shape[0], M.shape[1]
    mu = torch.full((N1,), -math.log(N1), dtype=M.dtype, device=M.device)
    nu = torch.full((N2,), -math.log(N2), dtype=M.dtype, device=M.device)

    if log:
        log = {"err": []}

    T  = sinkhorn(
                p,
                q,
                M,
                reg,
                niter= 50,
                warmstart=(mu, nu),
                mask=mask,
            )

    if abs(torch.sum(T) - 1) > 1e-5:
        warnings.warn(
            "Solver failed to produce a transport plan. You might "
            "want to increase the regularization parameter `epsilon`."
        )
    if semi_use:
        T = T[:,:-1]  # remove the last row which is all zeros
    else:
        T = T[:-1,:-1]  # remove the last column which is all zeros

    has_nan = torch.isnan(T).any() and torch.isinf(T).any()
    if has_nan:
        raise ValueError("Sinkhorn algorithm produced NaN values in the transport plan.")

    if log:
        log["wgw_dist"] = torch.sum(M * T)
        return T, log
    else:
        return T
