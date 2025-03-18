
# **Polar Code Analysis with Neural Networks**

This repository contains scripts and data for analyzing polar codes and comparing classical and neural network-based decoding approaches.

## **Overview**
Polar codes are a class of error-correcting codes that achieve the Shannon capacity under successive cancellation (SC) decoding. This project explores enhancements using neural networks for improved performance in unknown channels with and without memory.

## **Project Structure**
- **`classic_polar_coding.py`** - Implementation of the classic polar coding algorithm. Including functions for: random binary message creator, frozen bits addition, polar encoding, BPSK, AWGN,channel embedding, bit node and check node operations, SC polar decoding (given forzen indices), SC with genie (for analysis purposes).
- **`best_indices_selection.py`** - Identifies the best frozen bit positions for polar codes, by using Monte Carlo method, for AWGN, BEC, BSC channels. the output is json file with list of indices from least relaibel to best, per block length.
- **`frozen_bits.json`** - Precomputed sets of frozen bits for different block lengths. i.e the output of best_indices_selection.py
- **`polar_code_analysis.py`** - inputs: R(ration between info bits and frozen bits), channel type (AWGN/BEC/BSC), channel param (i.e noise), json file with indices rankings. Outputs - BER per block length, both print and plot.
- **`train_nn_embedding.py`** – Script for training the neural network (MINE) to estimate mutual information over an AWGN channel with BPSK-modulated inputs, and saving the trained model.
- **`test_nn_embedding.py`** - Evaluates the trained neural network's performance.
- **`nn_embedding.py`** – Defines the neural network model for computing Log-Likelihood Ratios (LLRs) using a trained MINE model for polar decoding.
- **`sc_nn_embedding.py`** - Implementation of SC decoding integrated with neural embeddings.
- **`nn_vs_classic_analysis.py`** - Compares the performance of neural network-based decoding against classical decoding methods.



# **Polar Code Analysis with Neural Networks**

This repository provides scripts and data for analyzing polar codes, comparing classical and neural network-based decoding methods to enhance performance, especially in unknown or memory channels.

## **Overview**
Polar codes are error-correcting codes capable of achieving Shannon capacity with successive cancellation (SC) decoding. This project investigates performance enhancements by integrating neural network approaches, particularly for channels with unknown characteristics.

## **Project Structure**

### Classic Polar Coding
- **`classic_polar_coding.py`** – Implements the classic polar coding algorithm, including:
  - Random binary message generation
  - Frozen bit insertion
  - Polar encoding
  - BPSK modulation
  - AWGN channel simulation
  - Channel embedding (LLR computation)
  - Check-node and bit-node operations
  - Successive Cancellation (SC) decoding given frozen indices
  - SC decoding with a genie for analysis purposes

### Frozen Bit Selection
- **`best_indices_selection.py`** – Determines optimal frozen bit positions for polar codes using Monte Carlo simulations across AWGN, BEC, and BSC channels. Outputs a JSON file ranking indices from least reliable to most reliable per block length.
- **`frozen_bits.json`** – Contains precomputed sets of frozen bits for various block lengths, generated by `best_indices_selection.py`.

### Analysis and Performance Evaluation
- **`polar_code_analysis.py`** – Computes and visualizes Bit Error Rate (BER) as a function of block length.
  - Inputs: Code rate (information bits vs. frozen bits), channel type (AWGN, BEC, BSC), channel parameters (e.g., noise level), and a JSON file of ranked indices.
  - Outputs: BER statistics and corresponding plots.

### Neural Network-Based Decoding
- **`train_nn_embedding.py`** – Trains a neural network (using the MINE framework) to estimate mutual information over an AWGN channel with BPSK modulation. The trained model is saved for inference.
- **`nn_embedding.py`** – Loads the trained MINE neural network model and computes Log-Likelihood Ratios (LLRs) used for polar decoding.
- **`test_nn_embedding.py`** – Evaluates the performance and accuracy of the trained neural network model.
- **`sc_nn_embedding.py`** – Implements Successive Cancellation decoding combined with neural network embeddings.

### Comparative Analysis
- **`nn_vs_classic_analysis.py`** – Compares the decoding performance of neural network-based methods against classical polar decoding methods, providing performance metrics and visual comparisons.


