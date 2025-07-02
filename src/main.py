import numpy as np

import torch
import torch.nn as nn

import random
import sys

# Nucleotide base or amino acid residue to index mapping, required for one-hot encoding of sequences
nucleotides_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

# Hyperparameters of 1D convolution module for capturing local k-mer patterns
conv1d_out_channels = 1; conv1d_kernel_size = 5

# Hyperparameters of projection submodule for expanded hash code generation
projection_dimension = 1000; sparsification_fraction = 0.1; do_binary_projection = False

# Hyperparameters of winner-take-all (WTA) thresholding submodule for sparse hash code generation
wta_fraction = 0.2; do_binary_wta = False

def generate_seq_hash_code(sequence: str, verbose: bool) -> np.ndarray:
    # Setting seeds of pseudo-random number generators for reusing same weight components
    np.random.seed(seed=42); torch.manual_seed(seed=42); random.seed(a=42)

    if verbose:
        print(f"generate_seq_hash_code: sequence.length = {len(sequence)}")
    
    # Generating one-hot encoded feature matrix of input sequence
    onehot_seq_hash = np.zeros(shape=(len(sequence), len(nucleotides_map)), dtype=np.float32)

    for seq_pos_idx, nucleotide in enumerate(sequence):
        if nucleotide in nucleotides_map:
            onehot_seq_hash[seq_pos_idx, nucleotides_map[nucleotide]] = 1
    
    if verbose:
        print(f"generate_seq_hash_code: onehot_seq_hash.shape = {onehot_seq_hash.shape}")
    
    # Applying 1D convolution to one-hot encoded feature matrix for capturing local motifs and producing feature map with reduced dimension
    conv1d = nn.Conv1d(in_channels=len(nucleotides_map), out_channels=conv1d_out_channels, kernel_size=conv1d_kernel_size)

    convoluted_seq_hash = conv1d(torch.from_numpy(onehot_seq_hash.T)).detach().numpy().T

    if verbose:
        print(f"generate_seq_hash_code: convoluted_seq_hash.shape = {convoluted_seq_hash.shape}")
    
    # Flattening feature map to obtain 1-dimensional feature vector
    flat_convoluted_seq_hash = np.reshape(a=convoluted_seq_hash, newshape=(convoluted_seq_hash.size,))

    if verbose:
        print(f"generate_seq_hash_code: flat_convoluted_seq_hash.shape = {flat_convoluted_seq_hash.shape}")
    
    # Generating projection matrix for eventual expansion of feature vector
    projection_matrix = None

    if do_binary_projection:
        projection_matrix = np.ones(shape=(projection_dimension, flat_convoluted_seq_hash.size), dtype=np.float32)
    else:
        projection_matrix = np.random.normal(loc=0, scale=1, size=(projection_dimension, flat_convoluted_seq_hash.size))
    
    for projection_dim_idx in range(projection_dimension):
        sparsification_indices = random.sample(population=range(flat_convoluted_seq_hash.size), k=int((1 - sparsification_fraction) * flat_convoluted_seq_hash.size))

        projection_matrix[projection_dim_idx, sparsification_indices] = 0
    
    wta_projected_seq_hash = projection_matrix @ flat_convoluted_seq_hash

    # Generating sparse expanded hash code with WTA mechanism
    wta_indices = np.argsort(a=wta_projected_seq_hash)[:: -1]

    wta_projected_seq_hash[wta_indices[int(wta_fraction * wta_projected_seq_hash.size):]] = 0

    if do_binary_wta:
        wta_projected_seq_hash[wta_indices[: int(wta_fraction * wta_projected_seq_hash.size)]] = 1
    
    if verbose:
        print(f"generate_seq_hash_code: wta_projected_seq_hash.shape = {wta_projected_seq_hash.shape}")
    
    return wta_projected_seq_hash

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Invalid number of command-line arguments provided!"
    
    input_seq_pairs_file_path = sys.argv[1]; num_seq_pairs = None

    with open(file=input_seq_pairs_file_path, mode='r') as input_seq_pairs_file:
        input_seq_pairs = [input_sequence for input_sequence in input_seq_pairs_file.read().split('\n') if input_sequence != '']

        num_seq_pairs = len(input_seq_pairs) // 2

        input_seq_pairs = [(input_seq_pairs[2 * seq_pair_idx], input_seq_pairs[2 * seq_pair_idx + 1]) for seq_pair_idx in range(num_seq_pairs)]
    
    distance_metric = 0; verbose = True

    for input_seq_pair in input_seq_pairs:
        seq1_hash_code = generate_seq_hash_code(sequence=input_seq_pair[0], verbose=verbose)

        verbose = False if verbose else verbose

        seq2_hash_code = generate_seq_hash_code(sequence=input_seq_pair[1], verbose=verbose)

        if do_binary_wta:
            hamming_distance = np.count_nonzero(a=seq1_hash_code != seq2_hash_code)

            distance_metric += hamming_distance / (projection_dimension * num_seq_pairs)
        else:
            normalized_seq1_hash_code = seq1_hash_code / np.sqrt(np.sum(np.power(seq1_hash_code, 2)))
            normalized_seq2_hash_code = seq2_hash_code / np.sqrt(np.sum(np.power(seq2_hash_code, 2)))

            cosine_similarity = normalized_seq1_hash_code @ normalized_seq2_hash_code

            distance_metric += cosine_similarity / projection_dimension
    
    if do_binary_wta:
        print(f"For {num_seq_pairs} sequence pairs, mean fractional Hamming distance is {distance_metric}")
    else:
        print(f"For {num_seq_pairs} sequence pairs, mean cosine similarity is {distance_metric}")