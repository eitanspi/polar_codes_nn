import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# Polar Code Functions
# ---------------------------
def sample(batch, k):
    """
    Generate a batch of k information bits.
    """
    b = np.random.randint(0, 2, size=(batch, k), dtype=np.uint8)
    return b


def rate_profile(b, f, N):
    """
    Place the information bits b into a length-N vector u with frozen bits (set to 0) at indices f.
    """
    u = [0] * N
    info_idx = 0
    for i in range(N):
        if i in f:
            u[i] = 0
        else:
            u[i] = b[info_idx]
            info_idx += 1
    return u


def encode(u):
    N = len(u)
    if N == 1:
        return u
    mid = N // 2
    left = encode(u[:mid])
    right = encode(u[mid:])
    return [left[i] ^ right[i] for i in range(mid)] + [right[i] for i in range(mid)]


def bpsk_modulate(x):
    """
    Map bits to BPSK symbols: 0 -> +1, 1 -> -1.
    """
    c = [1 - 2 * i for i in x]
    return c


def awgn(c, n0):
    """
    Pass the BPSK symbols through an AWGN channel.
    """
    c = np.array(c, dtype=float)
    sigma = np.sqrt(n0)
    noise = sigma * np.random.randn(len(c))
    y = c + noise
    y_clean = [round(i, 2) for i in y]
    return y_clean


def channel_embedding(y, n0):
    """
    Compute the Log-Likelihood Ratio (LLR) for each received symbol y.

    Parameters:
    - y: List or numpy array of received symbols after AWGN channel.
    - n0: Noise variance (2 * sigma^2).

    Returns:
    - llr: List of LLR values corresponding to each bit.

    Calculation:
    1. Given a transmitted BPSK symbol x ∈ {+1, -1}, the received signal y is:
         y = x + n,   where n ~ N(0, sigma^2) (Gaussian noise)

    2. The conditional probability density functions (PDFs) for y given x are:
         p(y | x = +1) = (1 / sqrt(2πσ²)) * exp(- (y - 1)² / (2σ²))
         p(y | x = -1) = (1 / sqrt(2πσ²)) * exp(- (y + 1)² / (2σ²))

    3. The LLR is defined as:
         LLR(y) = log( p(y | x = 0) / p(y | x = 1) )

    4. Substituting the Gaussian PDFs:
         LLR(y) = log( exp(- (y - 1)² / (2σ²)) / exp(- (y + 1)² / (2σ²)) )

    5. Simplifying:
         LLR(y) = - (y - 1)² / (2σ²) + (y + 1)² / (2σ²)
                = (2y) / (2σ²)
                = (2 / N0) * y    (since N0 = 2σ²)
    """
    llr = (2 / n0) * np.array(y)
    return llr


def check_node(r1, r2):
    """
    Compute the check–node LLR combining function.
    """
    return np.sign(r1) * np.sign(r2) * np.minimum(np.abs(r1), np.abs(r2))


def bit_node(r1, r2, u1):
    """
    Compute the bit–node LLR combining function.
    """
    return r2 + (1 - 2 * u1) * r1


def hard_dec(l):
    """
    Hard decision on an LLR: returns 0 if l >= 0, else 1.
    """
    return 0 if l >= 0 else 1


def interleave(arr):
    # Find the midpoint of the array
    N = len(arr)
    mid = N // 2

    # Create the result array by pairing elements
    result = [[arr[i], arr[i + mid]] for i in range(mid)]

    return result


def decode(x):
    """
    Inverse polar transform (for completeness).
    """
    u = encode(x)

    return u


def encode_u_arrays(u1, u2):
    """" Helper function for SC decoder """
    # Ensure both arrays have the same length
    if len(u1) != len(u2):
        raise ValueError("Arrays must have the same length")

    # Perform bitwise XOR for each pair of corresponding elements
    xor_result = [x ^ y for x, y in zip(u1, u2)]

    # Return the result as a flat array combining XOR results and u2
    return xor_result + u2


def sc_decoder(r_array, frozen_indices):
    index_counter = 0
    N = len(r_array)
    mid = N // 2

    def sc_decoder_in(r_array, index_counter):
        mid = len(r_array) // 2
        if len(r_array) == 1:
            if index_counter in frozen_indices:
                u_hat = 0
            else:
                u_hat = hard_dec(r_array[0])
            index_counter += 1
            return [u_hat], index_counter

        r_interleaved = interleave(r_array)

        left_array = [check_node(pair[0], pair[1]) for pair in r_interleaved]
        u_hat_left, index_counter = sc_decoder_in(left_array, index_counter)
        right_array = [bit_node(pair[0], pair[1], x_val) for pair, x_val in zip(r_interleaved, u_hat_left)]
        u_hat_right, index_counter = sc_decoder_in(right_array, index_counter)

        # Compute right_array
        x_hat = [u_hat_left[i] ^ u_hat_right[i] for i in range(mid)] + u_hat_right

        return x_hat, index_counter  # Ensure index_counter is returned and updated

    # Initial interleave and processing
    r_interleaved = interleave(r_array)
    r_interleaved = interleave(r_array)

    left_array = [check_node(pair[0], pair[1]) for pair in r_interleaved]
    u_hat_left, index_counter = sc_decoder_in(left_array, index_counter)
    right_array = [bit_node(pair[0], pair[1], x_val) for pair, x_val in zip(r_interleaved, u_hat_left)]
    u_hat_right, index_counter = sc_decoder_in(right_array, index_counter)

    # Compute right_array
    x_hat = [u_hat_left[i] ^ u_hat_right[i] for i in range(mid)] + u_hat_right
    u_hat = encode(x_hat)

    return u_hat


def sc_decoder_genie(r_array, u_array):
    """Checks if there is an error in decoding u[i] given that we decoded u[0],...,u[i-1] correctly"""
    errors_array = [0] * len(u_array)  # Initialize errors_array

    def sc_decoder_genie_in(r_array, u_array, index_counter):
        N = len(r_array)

        if N == 1:
            # If the received bit doesn't match the expected bit, mark an error
            if hard_dec(r_array[0]) != u_array[0]:
                errors_array[index_counter] = 1  # Update errors array
            return index_counter + 1  # Move index forward

        r_interleaved = interleave(r_array)

        # Compute left_array
        left_array = [check_node(pair[0], pair[1]) for pair in r_interleaved]

        # Compute right_array
        x_array = encode(u_array[:len(u_array) // 2])
        right_array = [bit_node(pair[0], pair[1], x_val) for pair, x_val in zip(r_interleaved, x_array)]

        # Recursive calls
        index_counter = sc_decoder_genie_in(left_array, u_array[:len(u_array) // 2], index_counter)
        index_counter = sc_decoder_genie_in(right_array, u_array[len(u_array) // 2:], index_counter)

        return index_counter  # Ensure index_counter is returned and updated

    # Initial interleave and processing
    r_interleaved = interleave(r_array)

    left_array = [check_node(pair[0], pair[1]) for pair in r_interleaved]
    x_array = encode(u_array[:len(u_array) // 2])
    right_array = [bit_node(pair[0], pair[1], x_val) for pair, x_val in zip(r_interleaved, x_array)]

    # Start recursion
    index_counter = 0
    index_counter = sc_decoder_genie_in(left_array, u_array[:len(u_array) // 2], index_counter)
    index_counter = sc_decoder_genie_in(right_array, u_array[len(u_array) // 2:], index_counter)

    return errors_array


