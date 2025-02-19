import numpy as np
import json
import matplotlib.pyplot as plt
import torch
from classic_polar_coding import sample, rate_profile, encode, bpsk_modulate, awgn, channel_embedding, sc_decoder
from sc_nn_embedding import nn_channel_embedding

# Load frozen bit indices from JSON
with open("frozen_bits_AWGN_N0_0.61.json", "r") as f:
    frozen_bits_dict = json.load(f)

# Set parameters
R = 0.25  # Code rate
channel_param = 0.61  # N0 for AWGN
plot_results = True

# Define block lengths N = 2^n
n_values = list(range(4, 11))  # n from 1 to 10
N_values = [2 ** n for n in n_values]
BER_classic = []
BER_nn = []

# Simulation parameters
num_trials = 5000  # Number of trials per block length
model_path = "mine_trained_on_N0_range.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

for N in N_values:
    K = int(R * N)  # Number of information bits
    frozen_indices = frozen_bits_dict[str(N)][:N - K]  # Get frozen bits
    errors_classic = 0
    errors_nn = 0
    total_bits = 0

    for _ in range(num_trials):
        # Generate random information bits
        b = sample(1, K)[0]

        # Encode
        u = rate_profile(b, frozen_indices, N)
        x = encode(u)

        # Modulate
        c = bpsk_modulate(x)

        # Pass through AWGN channel
        y = awgn(c, channel_param)

        # Compute LLRs
        llr_classic = channel_embedding(y, channel_param)
        llr_nn = nn_channel_embedding(y, model_path, device).tolist()

        # Decode using SC decoder
        u_hat_classic = sc_decoder(llr_classic, frozen_indices)
        u_hat_nn = sc_decoder(llr_nn, frozen_indices)

        # Compute errors
        errors_classic += np.sum(np.array(u_hat_classic) != np.array(u))
        errors_nn += np.sum(np.array(u_hat_nn) != np.array(u))
        total_bits += N

    BER_classic.append(errors_classic / total_bits)
    BER_nn.append(errors_nn / total_bits)
    print(f"N={N}, BER_Classic={BER_classic[-1]:.5e}, BER_NN={BER_nn[-1]:.5e}")

# Plot results
if plot_results:
    plt.figure()
    plt.semilogy(n_values, BER_classic, marker='o', linestyle='-', label="Classic Embedding")
    plt.semilogy(n_values, BER_nn, marker='s', linestyle='-', label="NN Embedding")
    plt.xlabel("n (Block Length N=2^n)")
    plt.ylabel("BER (Bit Error Rate)")
    plt.title("Polar Code Performance: Classic vs NN Embedding")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.show()
