# attacks.py
# PGD L_inf implementation with restarts (torch)
import torch
#torch.nn.functional allows us to access cross entropy function so we dont have to make it ourselves
import torch.nn.functional as F
import numpy as np
from tqdm import trange

def pgd_linf(model, x, y_true, eps=0.02, step=None, iters=100, restarts=5, device=None):
    """
    Perform a PGD vector space attack which generates an adverserial example (slightly modified input inmage that causes the model to misclassify while staying within a bounded pixel range - set by epsilon)

    model:      PyTorch model (in eval mode)
    x:          input tensor shape (1,C,H,W)
    y_true:     int label (or tensor)
    eps:        L_inf radius
    step:       step size (if None use eps/10)
    iters:      iterations per restart
    restarts:   restarts count
    Returns:    (found (bool), adv_tensor_or_None)
    """
    if device is None:
        device = x.device
    model.to(device).eval()
    x_orig = x.clone().to(device)
    y = torch.tensor([y_true], device=device)

    #default gradient step per iteration is one-tenth of epsilon
    if step is None:
        step = eps / 10.0

    best_adv = None         #placeholder if no attack succeeds

    for r in range(restarts):
        # start from random perturbation within box to improve robusteness
        delta = torch.empty_like(x_orig).uniform_(-eps, eps).to(device)     #initialize random noise tensor within vector space bounds
        adv = torch.clamp(x_orig + delta, 0.0, 1.0).detach()                #ensures validity
        adv.requires_grad = True        #compute grads with respect to adverserial image

        for i in range(iters):
            logits = model(adv)                     #forward pass (compute for current adverserial image)
            loss = F.cross_entropy(logits, y)       #compute cross entropy loss wrt true label (maximize to make model wrong)
            loss.backward()                         #backpropagate
            # gradient step - move in direction that INCREASES LODD
            grad = adv.grad.data
            adv = adv + step * torch.sign(grad)     #ensure update still in vector space
            # project back into L_inf ball - ensure in bounds again
            adv = torch.max(torch.min(adv, x_orig + eps), x_orig - eps)
            #break computation graph for next iteraion with detatch
            adv = torch.clamp(adv, 0.0, 1.0).detach()
            adv.requires_grad = True
            # early stop if misclassified
            with torch.no_grad():
                pred = model(adv).argmax(dim=1).item()
                if pred != int(y_true):
                    #immediately return success and adverserial image
                    return True, adv.detach().cpu()
                
    #no adverserial image found
    return False, None
