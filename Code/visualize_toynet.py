'''
Background colors: predicted class regions from trained network.

Training points: the original dataset points.

Yellow dot: the test input you are verifying.

Green square: the ε-ball around your test input.

Red dot: adversarial point found by Z3 (if any).
'''

import matplotlib.pyplot as plt
import numpy as np
import torch
from train_toy_network import TinyNet, X_train, y_train, scaler

def plot_decision_boundary(model, X, y, test_point=None, eps=0.05, adv_point=None):
    """
    Plot decision boundary of a 2D network, training points, test point,
    epsilon ball, and adversarial point (if any).
    """
    # Create grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_t = torch.tensor(grid, dtype=torch.float32)
    
    # Predict class for each grid point
    with torch.no_grad():
        preds = model(grid_t).argmax(dim=1).numpy()
    preds = preds.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, preds, alpha=0.3, cmap=plt.cm.coolwarm)
    
    # Plot training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor='k', s=40, label="Train")
    
    # Plot test point
    if test_point is not None:
        plt.scatter(test_point[0], test_point[1], c='yellow', edgecolor='k', s=100, label='Test point')
        # Draw epsilon L∞ ball
        rect = plt.Rectangle((test_point[0]-eps, test_point[1]-eps),
                             2*eps, 2*eps, linewidth=2, edgecolor='green', facecolor='none', label='ε-ball')
        plt.gca().add_patch(rect)
    
    # Plot adversarial point
    if adv_point is not None:
        plt.scatter(adv_point[0], adv_point[1], c='red', edgecolor='k', s=100, label='Adversarial')
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Decision Boundary + ε-ball + Adversarial Example')
    plt.legend()
    plt.show()

# Example usage:
# plot_decision_boundary(model, X_train, y_train, test_point=x0, eps=EPS, adv_point=counter_x if res==sat else None)
