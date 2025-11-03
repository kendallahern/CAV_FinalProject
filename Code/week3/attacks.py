# attacks.py
# PGD L_inf implementation with restarts (torch)
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange

def pgd_linf(model, x, y_true, eps=0.02, step=None, iters=100, restarts=5, device=None):
    """
    model: PyTorch model (in eval mode)
    x: input tensor shape (1,C,H,W)
    y_true: int label (or tensor)
    eps: L_inf radius
    step: step size (if None use eps/10)
    iters: iterations per restart
    restarts: restarts count
    Returns: (found (bool), adv_tensor_or_None)
    """
    if device is None:
        device = x.device
    model.to(device).eval()
    x_orig = x.clone().to(device)
    y = torch.tensor([y_true], device=device)

    if step is None:
        step = eps / 10.0

    best_adv = None
    for r in range(restarts):
        # start from random perturbation within box
        delta = torch.empty_like(x_orig).uniform_(-eps, eps).to(device)
        adv = torch.clamp(x_orig + delta, 0.0, 1.0).detach()
        adv.requires_grad = True
        for i in range(iters):
            logits = model(adv)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            # gradient step
            grad = adv.grad.data
            adv = adv + step * torch.sign(grad)
            # project back into L_inf ball
            adv = torch.max(torch.min(adv, x_orig + eps), x_orig - eps)
            adv = torch.clamp(adv, 0.0, 1.0).detach()
            adv.requires_grad = True
            # early stop if misclassified
            with torch.no_grad():
                pred = model(adv).argmax(dim=1).item()
                if pred != int(y_true):
                    return True, adv.detach().cpu()
    return False, None
