import numpy as np
import json
import matplotlib.pyplot as plt
from classic_polar_coding import sample, rate_profile, encode, bpsk_modulate, awgn, channel_embedding, sc_decoder

# Load frozen bit indices from JSON
with open("frozen_bits_AWGN_N0_0.61.json", "r") as f:
    frozen_bits_dict = json.load(f)

# Set parameters
R = 0.25  # Code rate
channel_type = "AWGN"  # Choose from "AWGN", "BSC", "BEC"
channel_param = 0.8  # N0 for AWGN, flip probability for BSC, erasure probability for BEC
plot_results = True

# Define block lengths N = 2^n
n_values = list(range(1, 11))  # n from 1 to 12
N_values = [2 ** n for n in n_values]
BER_results = []

# Simulation parameters
num_trials = 5000  # Number of trials per block length

for N in N_values:
    K = int(R * N)  # Number of information bits
    frozen_indices = frozen_bits_dict[str(N)][:N - K]  # Get frozen bits
    errors = 0
    total_bits = 0

    for _ in range(num_trials):
        # Generate random information bits
        b = sample(1, K)[0]

        # Encode
        u = rate_profile(b, frozen_indices, N)
        x = encode(u)

        # Modulate
        c = bpsk_modulate(x)

        # Pass through channel
        if channel_type == "AWGN":
            y = awgn(c, channel_param)
            llr = channel_embedding(y, channel_param)
        elif channel_type == "BSC":
            y = np.array(c)
            flips = np.random.rand(len(y)) < channel_param
            y[flips] *= -1
            llr = 2 * y  # Hard LLR for BSC
        elif channel_type == "BEC":
            y = np.array(c)
            erasures = np.random.rand(len(y)) < channel_param
            llr = np.where(erasures, 0, 2 * y)  # Erasures get LLR = 0
        else:
            raise ValueError("Invalid channel type")

        # Decode
        u_hat = sc_decoder(llr, frozen_indices)

        # Compute errors
        errors += np.sum(np.array(u_hat) != np.array(u))
        total_bits += N

    BER = errors / total_bits
    BER_results.append(BER)
    print(f"N={N}, BER={BER:.5e}")

# Plot results
if plot_results:
    plt.figure()
    plt.semilogy(n_values, BER_results, marker='o', linestyle='-', label=f"{channel_type} (param={channel_param})")
    plt.xlabel("n (Block Length N=2^n)")
    plt.ylabel("BER (Bit Error Rate)")
    plt.title("Polar Code Performance")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.show()
