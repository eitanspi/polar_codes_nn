#!/usr/bin/env python3
"""
eval_mine.py

Loads a trained MINE model, generates test samples at a chosen N0,
computes the true embedding E(y) = 2y / N0 and the NN-estimated
embedding E_NN(y) = T(1,y) - T(0,y). Prints pairs and optionally
plots them.

Author: Your Name
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------
# 1) Define the same MINE network architecture
# -----------------------------------------------------
class MINE(nn.Module):
    def __init__(self):
        super(MINE, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, y):
        input_xy = torch.stack([x.float(), y.float()], dim=1)
        return self.net(input_xy)


# -----------------------------------------------------
# 2) Utility: generate test data at fixed N0
# -----------------------------------------------------
def generate_test_data(n_samples, N0, device="cpu"):
    """
    Generate test samples for X ~ Bernoulli(0.5), Y = X_bpsk + noise,
    with a fixed N0.
    """
    X = torch.randint(0, 2, (n_samples,), device=device)
    X_bpsk = 1 - 2 * X  # BPSK in {-1,1}
    noise_std = torch.sqrt(torch.tensor([N0], device=device))
    noise = torch.randn(n_samples, device=device) * noise_std
    Y = X_bpsk + noise
    return X, Y


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) Instantiate model and load weights
    model = MINE()
    model_path = "mine_trained_on_N0_range.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    print(f"Loaded model from {model_path}")

    # 2) Generate test data at a chosen N0
    #    (You can change this to test various noise levels)
    N0_test = 0.5
    n_samples = 1000
    X, Y = generate_test_data(n_samples, N0_test, device=device)

    # 3) Compute the true embedding E(y) = 2y / N0
    #    We'll do it for each Y in the test set.
    E_true = (2 * Y) / N0_test

    # 4) Compute the NN-estimated embedding:
    #    E_NN(y) = T(1,y) - T(0,y)
    #    We'll treat '1' and '0' as separate calls.
    #    We'll do a forward pass for X=1 for all Y, and X=0 for all Y.
    #    So we create two separate inputs:
    ones = torch.ones_like(Y, device=device)
    zeros = torch.zeros_like(Y, device=device)

    T_1y = model(ones, Y).squeeze()  # shape [n_samples]
    T_0y = model(zeros, Y).squeeze()  # shape [n_samples]
    # E_est = T_1y - T_0y
    # E_est = - E_est
    E_est = T_0y - T_1y

    # 5) Print the first 100 pairs (E_true, E_est)
    print("\nFirst 100 pairs of (E_true, E_est):")
    for i in range(min(100, n_samples)):
        print(f"{i:3d}) E_true={E_true[i].item():.4f}, E_est={E_est[i].item():.4f}")

    # 6) Plot E_true vs E_est
    E_true_np = E_true.detach().cpu().numpy()
    E_est_np = E_est.detach().cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(E_true_np, E_est_np, alpha=0.5, s=10)
    plt.plot([-10, 10], [-10, 10], 'r--')  # ideal line
    plt.xlim([E_true_np.min() - 1, E_true_np.max() + 1])
    plt.ylim([E_est_np.min() - 1, E_est_np.max() + 1])
    plt.xlabel("True Embedding E(y)")
    plt.ylabel("Estimated Embedding E_NN(y)")
    plt.title(f"E_true vs E_est, N0={N0_test}")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
