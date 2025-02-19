#!/usr/bin/env python3
"""
train_mine.py

Trains a neural network T(X, Y) using the MINE (Donsker-Varadhan) objective
over an AWGN channel with BPSK-modulated inputs X. Saves the trained model.

Author: Your Name
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# -------------------------
# 1) Define the neural network
# -------------------------
class MINE(nn.Module):
    """
    A simple MLP that estimates T(X,Y).
    Input: (X, Y) where X is in {0,1} and Y is a float.
    Output: T(X, Y) as a single scalar.
    """
    def __init__(self):
        super(MINE, self).__init__()
        # Example architecture: 3 hidden layers of size 128
        self.net = nn.Sequential(
            nn.Linear(2, 128),  # Input is 2D: [X, Y]
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)   # Output is scalar T(X,Y)
        )

    def forward(self, x, y):
        """
        x: shape [batch_size], elements in {0,1}
        y: shape [batch_size]
        Returns scalar T(x_i, y_i) for each sample in the batch.
        """
        # Concatenate X and Y into shape [batch_size, 2]
        input_xy = torch.stack([x.float(), y.float()], dim=1)
        return self.net(input_xy)


# -------------------------
# 2) Data generation function
# -------------------------
def generate_data(batch_size, device, n0_min=0.1, n0_max=1.0):
    """
    Generate real pairs (X, Y) ~ P(X,Y) and negative pairs (X', Y) ~ P(X)P(Y).

    X ~ Bernoulli(0.5), X_BPSK = 1 - 2*X in {-1,1}
    Y = X_BPSK + noise, noise ~ N(0, N0)

    We sample N0 uniformly in [n0_min, n0_max] for each sample,
    for increased generalization.
    """

    # Sample X in {0, 1}
    X = torch.randint(0, 2, (batch_size,), device=device)
    # Convert to BPSK in {-1, 1}
    X_bpsk = 1 - 2*X  # if X=0 => 1, if X=1 => -1

    # Sample noise variance N0 uniformly
    N0 = torch.rand(batch_size, device=device) * (n0_max - n0_min) + n0_min
    noise_std = torch.sqrt(N0)  # standard deviation

    # Generate AWGN
    noise = torch.randn(batch_size, device=device) * noise_std

    # Y = X_bpsk + noise
    Y = X_bpsk + noise

    # Positive samples (X, Y)
    X_pos = X
    Y_pos = Y

    # Negative samples: shuffle X to break correlation with Y
    # This approximates sampling from P(X)P(Y).
    X_neg = X[torch.randperm(batch_size)]

    # Return (X_pos, Y_pos), (X_neg, Y_pos), plus N0 for reference if needed
    return X_pos, Y_pos, X_neg, Y_pos, N0


# -------------------------
# 3) Training loop
# -------------------------
def train_mine(
    model,
    device="cpu",
    epochs=50,
    batch_size=1024,
    lr=1e-3,
    n0_min=0.1,
    n0_max=1.0
):
    """
    Trains the MINE model using the DV-based loss:

        I(X;Y) ~ E[T(X,Y)] - log( E[e^{T(X',Y)}] )

    model: MINE neural network
    device: "cpu" or "cuda"
    epochs: number of training epochs
    batch_size: batch size for training
    lr: learning rate
    n0_min, n0_max: range for noise variance N0
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # We'll do a simple training loop
    for epoch in range(epochs):
        # Generate data
        X_pos, Y_pos, X_neg, Y_neg, _ = generate_data(
            batch_size, device, n0_min, n0_max
        )

        # Forward pass on real pairs
        T_pos = model(X_pos, Y_pos)  # shape [batch_size, 1]

        # Forward pass on negative pairs
        T_neg = model(X_neg, Y_neg)

        # MINE loss = -( E[T_pos] - log(E[exp(T_neg)]) )
        # We'll compute these expectations as simple means over the batch.
        E_T_pos = T_pos.mean()
        # Compute mean of exp(T_neg)
        exp_T_neg = torch.exp(T_neg)
        mean_exp_T_neg = exp_T_neg.mean()

        loss = -(E_T_pos - torch.log(mean_exp_T_neg + 1e-12))

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] | "
                f"Loss: {loss.item():.4f} | "
                f"E[T_pos]: {E_T_pos.item():.4f} | "
                f"mean(exp(T_neg)): {mean_exp_T_neg.item():.4f}"
            )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) Instantiate the MINE network
    model = MINE()

    # 2) Train
    train_mine(
        model,
        device=device,
        epochs=1000,        # adjust as needed
        batch_size=1024,  # adjust as needed
        lr=1e-3,
        n0_min=0.1,
        n0_max=1.0
    )

    # 3) Save the trained model
    model_filename = "mine_trained_on_N0_range.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to {model_filename}")


if __name__ == "__main__":
    main()

"""
train_improved_mine.py

Trains an improved neural network T(X, Y) using the MINE (Donsker-Varadhan) objective
over an AWGN channel with BPSK-modulated inputs X. Saves the trained model.

The improved model features:
  - Increased hidden dimensionality (256 units per layer)
  - Batch normalization for training stability
  - Dropout for regularization
  - Kaiming weight initialization
"""

