import numpy as np

import torch
import torch.nn as nn

import random

nucleotides_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

np.random.seed(seed=42)
torch.manual_seed(seed=42)

conv1d_out_channels = 1
conv1d_kernel_size = 5

projection_dimension = 1000
sparsification_fraction = 0.1
wta_fraction = 0.2

do_binary_projection = False
do_binary_wta = False

if __name__ == "__main__":
    dna_seq = "ACGTAGCTACGTATGC"
    
    onehot_seq_hash = np.zeros(shape=(len(dna_seq), len(nucleotides_map)), dtype=np.float32)

    for seq_idx, nucleotide in enumerate(dna_seq):
        if nucleotide in nucleotides_map:
            onehot_seq_hash[seq_idx, nucleotides_map[nucleotide]] = 1
    
    conv1d = nn.Conv1d(in_channels=len(nucleotides_map), out_channels=conv1d_out_channels, kernel_size=conv1d_kernel_size)

    conv_seq_hash = conv1d(torch.from_numpy(onehot_seq_hash.T)).detach().numpy().T

    flat_conv_seq_hash = np.reshape(a=conv_seq_hash, newshape=(conv_seq_hash.size,))

    projection_matrix = None

    if do_binary_projection:
        projection_matrix = np.ones(shape=(projection_dimension, flat_conv_seq_hash.size), dtype=np.float32)
    else:
        projection_matrix = np.random.normal(loc=0, scale=1, size=(projection_dimension, flat_conv_seq_hash.size))

    for projection_idx in range(projection_dimension):
        sparsification_indices = random.sample(population=range(flat_conv_seq_hash.size), k=int((1 - sparsification_fraction) * flat_conv_seq_hash.size))

        projection_matrix[projection_idx, sparsification_indices] = 0
    
    wta_projected_seq_hash = projection_matrix @ flat_conv_seq_hash

    wta_indices = np.argsort(a=wta_projected_seq_hash)[:: -1][int(wta_fraction * wta_projected_seq_hash.size):]

    wta_projected_seq_hash[wta_indices] = 0

    if do_binary_wta:
        wta_indices = np.argsort(a=wta_projected_seq_hash)[:: -1][: int(wta_fraction * wta_projected_seq_hash.size)]

        wta_projected_seq_hash[wta_indices] = 1
    
    print(wta_projected_seq_hash)