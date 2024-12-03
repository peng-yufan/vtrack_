from lib.utils.box_ops import *
import torch
from einops import rearrange

def KL_loss(p, q):
    """
    p/q
    """
    avgp, covp = p
    avgq, covq = q
    B, G, C = avgp.shape
    covq_inv = covq.float().inverse()
    loss = (avgp-avgq).view(B, G, 1, C) @ covq_inv @ (avgp-avgq).view(B, G, C, 1)
    loss = loss.view(B, G)
    mat = covq_inv @ covp
    loss -= torch.log(torch.det(mat.float()))
    for i in range(C):
        loss += mat[:, :, i, i]
    loss -= C
    loss *= 0.5
    return torch.mean(loss)

def gauss_distribution(x, num_group=24, order=None):
    """
    x: B,N,C
    """
    if order == None:
        order = torch.randperm(768).cuda()

    B, N, C = x.shape
    x = x[:, :, order]
    # x = x.float()
    avg = torch.mean(x, dim=-2)
    x = x-avg.view(B, 1, -1)
    x = rearrange(x, 'b n (g c) -> b g n c', g=num_group)
    cov = x.permute([0, 1, 3, 2]) @ x/N
    avg = rearrange(avg, 'b (g c) -> b g c', g=num_group)
    # cov = x.permute([0, 2, 1]) @ x/N-avg.view(B, C, 1)@avg.view(B, 1, C)
    return avg, cov