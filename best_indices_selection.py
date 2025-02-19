from classic_polar_coding import *
import json
import numpy as np


def simulate_awgn_and_rank(N, n0, num_trials=10000):
    error_counts = np.zeros(N)
    for _ in range(num_trials):
        u = np.random.randint(0, 2, size=N)
        x = encode(u)
        c = bpsk_modulate(x)
        y = awgn(c, n0)
        llr = channel_embedding(y, n0)
        errors = sc_decoder_genie(llr, u)
        error_counts += np.array(errors)
    return np.argsort(-error_counts).tolist()


def simulate_bsc_and_rank(N, p, num_trials=10000):
    error_counts = np.zeros(N)
    for _ in range(num_trials):
        u = np.random.randint(0, 2, size=N)
        x = encode(u)
        c = bpsk_modulate(x)
        y = np.array(c)
        flips = np.random.rand(len(y)) < p
        y[flips] *= -1  # Flip bits with probability p
        llr = channel_embedding(y, 2 * (1 - 2 * p) / (p * (1 - p)))
        errors = sc_decoder_genie(llr, u)
        error_counts += np.array(errors)
    return np.argsort(-error_counts).tolist()


def simulate_bec_and_rank(N, p, num_trials=10000):
    error_counts = np.zeros(N)
    for _ in range(num_trials):
        u = np.random.randint(0, 2, size=N)
        x = encode(u)
        c = bpsk_modulate(x)
        y = np.array(c)
        erasures = np.random.rand(len(y)) < p
        y[erasures] = 0  # Erasure modeled as 0
        llr = channel_embedding(y, 1e6)  # High LLR for erased bits
        errors = sc_decoder_genie(llr, u)
        error_counts += np.array(errors)
    return np.argsort(-error_counts).tolist()


def generate_frozen_bit_files(n_range=(1, 11), channel_type="AWGN", param=0.61, num_trials=5000):
    frozen_bits_dict = {}
    for n in range(n_range[0], n_range[1] + 1):
        N = 2 ** n
        if channel_type == "AWGN":
            reliability_ranking = simulate_awgn_and_rank(N, param, num_trials)
            file_name = f"frozen_bits_AWGN_N0_{param}.json"
        elif channel_type == "BSC":
            reliability_ranking = simulate_bsc_and_rank(N, param, num_trials)
            file_name = f"frozen_bits_BSC_p_{param}.json"
        elif channel_type == "BEC":
            reliability_ranking = simulate_bec_and_rank(N, param, num_trials)
            file_name = f"frozen_bits_BEC_p_{param}.json"
        else:
            raise ValueError("Invalid channel type. Choose from 'AWGN', 'BSC', or 'BEC'.")

        frozen_bits_dict[str(N)] = reliability_ranking

    with open(file_name, "w") as f:
        json.dump(frozen_bits_dict, f, indent=4)

    print(f"Frozen bit indices saved to {file_name}")

# Example Usage
generate_frozen_bit_files(channel_type="AWGN", param=0.61)
# generate_frozen_bit_files(channel_type="BSC", param=0.1)
# generate_frozen_bit_files(channel_type="BEC", param=0.2)
