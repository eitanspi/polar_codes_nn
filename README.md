
# **Polar Code Analysis with Neural Networks**

This repository contains scripts and data for analyzing polar codes and comparing classical and neural network-based decoding approaches.

## **Overview**
Polar codes are a class of error-correcting codes that achieve the Shannon capacity under successive cancellation (SC) decoding. This project explores enhancements using neural networks for improved performance in unknown channels with and without memory.

## **Project Structure**
- **`classic_polar_coding.py`** - Implementation of the classic polar coding algorithm.
- **`polar_code_analysis.py`** - Tools for comprehensive analysis of polar codes.
- **`nn_embedding.py`** - Defines the neural network model for decoding polar codes.
- **`train_nn_embedding.py`** - Script for training the neural network model.
- **`test_nn_embedding.py`** - Evaluates the trained neural network's performance.
- **`sc_nn_embedding.py`** - Implementation of SC decoding integrated with neural embeddings.
- **`best_indices_selection.py`** - Identifies the best frozen bit positions for polar codes.
- **`nn_vs_classic_analysis.py`** - Compares the performance of neural network-based decoding against classical decoding methods.
- **`frozen_bits.json`** - Precomputed sets of frozen bits for different block lengths.



