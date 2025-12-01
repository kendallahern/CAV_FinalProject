# marabou_verify_point.py
from maraboupy import Marabou, MarabouCore
import numpy as np
import torch
from MiniNet import MiniNet

def marabou_verify_point(
    onnx_path,
    model_pth,
    x0,
    eps,
    delta=0.0,
    timeout=60,
    single_branch=False
):
    """
    onnx_path: path to MiniNet.onnx
    model_pth: path to MiniNet.pth
    x0: numpy array (1,28,28)
    eps: float
    delta: misclassification margin
    timeout: seconds
    single_branch: only test one alternative class (much faster)
    """

    # ---------------------------
    # Load PyTorch model for center label
    # ---------------------------
    model = MiniNet()
    model.load_state_dict(torch.load(model_pth, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(x0).unsqueeze(0).float())
        pred = int(torch.argmax(logits, dim=1).item())

    # ---------------------------
    # Load ONNX into Marabou
    # ---------------------------
    network = Marabou.read_onnx(onnx_path)

    inputVars = network.inputVars[0].flatten()
    outputVars = network.outputVars[0].flatten()
    xflat = x0.flatten()

    # ---------------------------
    # Apply eps bounds
    # ---------------------------
    eps_val = eps
    for i, v in enumerate(inputVars):
        lo = max(0.0, float(xflat[i] - eps_val))
        hi = min(1.0, float(xflat[i] + eps_val))
        network.setLowerBound(v, lo)
        network.setUpperBound(v, hi)

    # ---------------------------
    # Disjunction: output[j] >= output[pred] + delta
    # ---------------------------
    disjunction = []

    classes_to_test = (
        [(pred + 1) % 10] if single_branch else [j for j in range(10) if j != pred]
    )

    for j in classes_to_test:
        eq = MarabouCore.Equation(MarabouCore.Equation.GE)
        # Handle scalar vs array safely
        out_j_var = int(outputVars[j]) if np.isscalar(outputVars[j]) else int(outputVars[j][0])
        out_pred_var = int(outputVars[pred]) if np.isscalar(outputVars[pred]) else int(outputVars[pred][0])
        eq.addAddend(1.0, out_j_var)
        eq.addAddend(-1.0, out_pred_var)
        eq.setScalar(delta)
        disjunction.append([eq])

    network.addDisjunctionConstraint(disjunction)

    # ---------------------------
    # Solve
    # ---------------------------
    options = Marabou.createOptions(timeoutInSeconds=timeout)
    vals, stats = network.solve(options=options)

    result = {
        "result": "sat" if vals else "unsat",
        "time_s": stats.getTotalTimeInSeconds()
    }

    if vals:
        ce = [vals[v] for v in inputVars]
        result["counterexample"] = ce

    return result
