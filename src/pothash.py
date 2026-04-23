from typing import Callable
from dataclasses import dataclass, field
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import Optional
import random
from typing import Sequence
import torch
from typing import Tuple

dna_alphabet = ['A', 'C', 'G', 'T']

# Utility functions
def encode_kmer_to_int(kmer: str) -> int:
    encoded_kmer = 0

    for base in kmer:
        encoded_kmer = encoded_kmer * 8

        # If the base is not in the alphabet, set it to len(dna_alphabet) = 4 (unknown)
        encoded_kmer = encoded_kmer + (dna_alphabet.index(base) if base in dna_alphabet else len(dna_alphabet))
    
    return encoded_kmer

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_sequence(sequence: str, should_convert_rna_to_dna: bool = True) -> str:
    sequence = sequence.strip().upper()

    if should_convert_rna_to_dna:
        sequence = sequence.replace('U', 'T')

    return ''.join(base if base in dna_alphabet else 'N' for base in sequence)

def set_seeds_globally_for_reproducibility(seed: int = 42) -> None:
    np.random.seed(seed=seed)
    random.seed(a=seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    return None

# Encoders
class OneHotEncoder(nn.Module):
    def __init__(self, alphabet: list = dna_alphabet):
        super().__init__()

        self.alphabet = alphabet
        self.num_channels = len(alphabet)
    
    def forward(self, sequence: str) -> torch.Tensor:
        sequence = preprocess_sequence(sequence=sequence)

        one_hot_encoding = torch.zeros(size=(1, self.num_channels, len(sequence)), dtype=torch.float32)

        for position, base in enumerate(sequence):
            if base in self.alphabet:
                # Unknown bases become all-zero columns
                one_hot_encoding[0, self.alphabet.index(base), position] = 1.0
        
        return one_hot_encoding

class LearnedEmbeddingEncoder(nn.Module):
    def __init__(self, alphabet: list = dna_alphabet, embedding_dim: int = 8):
        super().__init__()

        self.alphabet = alphabet

        # One additional embedding for unknown bases
        self.embedding_table = nn.Embedding(num_embeddings=len(alphabet) + 1, embedding_dim=embedding_dim)
    
    def _convert_sequence_to_indices(self, sequence: str) -> torch.Tensor:
        sequence = preprocess_sequence(sequence=sequence)

        base_indices = []

        for base in sequence:
            if base in self.alphabet:
                base_indices.append(self.alphabet.index(base))
            else:
                # Unknown bases are mapped to the last index and the model will learn around it
                base_indices.append(len(self.alphabet))

        return torch.tensor(base_indices, dtype=torch.uint8)
    
    def forward(self, sequence: str) -> torch.Tensor:
        base_indices = self._convert_sequence_to_indices(sequence=sequence)

        embedding = self.embedding_table(base_indices)

        return embedding.transpose(0, 1).unsqueeze(0)

# Submer selection
class MinimizerMasker:
    def __init__(self, kmer_size: int = 15, window_size: int = 20):
        if kmer_size <= 0 or window_size <= 0:
            raise ValueError("kmer_size and window_size must be > 0")
        
        if kmer_size > window_size:
            raise ValueError("kmer_size must be <= window_size")
        
        self.kmer_size = kmer_size
        self.window_size = window_size

    def __call__(self, sequence: str, hash_function: Callable[[str], int]) -> torch.Tensor:
        sequence = preprocess_sequence(sequence=sequence)

        minimizer_mask = torch.zeros(size=(len(sequence),), dtype=torch.uint8)

        if len(sequence) < self.kmer_size:
            # If the sequence is shorter than the kmer size, return an empty mask
            return minimizer_mask
        
        kmer_hash_values = []

        for kmer_starting_position in range(len(sequence) - self.kmer_size + 1):
            kmer = sequence[kmer_starting_position: kmer_starting_position + self.kmer_size]

            if 'N' in kmer:
                kmer_hash_values.append(None)
            else:
                kmer_hash_values.append(hash_function(kmer))
        
        num_kmers_per_window = self.window_size - self.kmer_size + 1

        if len(kmer_hash_values) < num_kmers_per_window:
            # If the sequence is shorter than the window size, return an empty mask
            return minimizer_mask
        
        best_kmer_starting_positions_seen_so_far = set()
        
        for window_starting_position in range(len(kmer_hash_values) - num_kmers_per_window + 1):
            kmer_hash_values_in_window = kmer_hash_values[window_starting_position: window_starting_position + num_kmers_per_window]

            valid_kmer_hash_values_in_window = [(window_starting_position + kmer_idx, kmer_hash_value) for kmer_idx, kmer_hash_value in enumerate(kmer_hash_values_in_window) if kmer_hash_value is not None]

            if len(valid_kmer_hash_values_in_window) == 0:
                # If there are no valid kmer hash values in the window, skip this window
                continue

            best_kmer_starting_position, _ = min(valid_kmer_hash_values_in_window, key=lambda kmer_hash_position_value: kmer_hash_position_value[1])

            if best_kmer_starting_position not in best_kmer_starting_positions_seen_so_far:
                best_kmer_starting_positions_seen_so_far.add(best_kmer_starting_position)

                minimizer_mask[best_kmer_starting_position: best_kmer_starting_position + self.kmer_size] = 1
        
        return minimizer_mask

# Multi-scale convolution block
class MultiScaleConvolutionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 8, branch_kernel_sizes: Sequence[int] = (3, 5, 9), should_use_batchnorm: bool = True):
        super().__init__()

        if len(branch_kernel_sizes) == 0:
            raise ValueError("branch_kernel_sizes must be non-empty")
        
        self.should_use_batchnorm = should_use_batchnorm

        self.convolution_branches = nn.ModuleList()
        self.batchnorms = nn.ModuleList()

        for kernel_size in branch_kernel_sizes:
            padding = kernel_size // 2

            self.convolution_branches.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=False))

            if should_use_batchnorm:
                self.batchnorms.append(nn.BatchNorm1d(num_features=out_channels))
        
        self.out_channels = out_channels * len(branch_kernel_sizes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        extracted_features = []

        for convolution_branch_idx, convolution_block in enumerate(self.convolution_branches):
            y = convolution_block(x)

            if self.should_use_batchnorm:
                y = self.batchnorms[convolution_branch_idx](y)
            
            y = F.relu(input=y)

            extracted_features.append(y)
        
        return torch.cat(tensors=extracted_features, dim=1)

# Pooling and normalization
class PoolingAndNormalization(nn.Module):
    def __init__(self, pooling_mode: str = "max_mean", should_l2_normalize: bool = True):
        super().__init__()

        if pooling_mode not in {"max", "mean", "max_mean"}:
            raise ValueError("pooling_mode must be one of {'max', 'mean', 'max_mean'}")
        
        self.pooling_mode = pooling_mode
        self.should_l2_normalize = should_l2_normalize
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled_features = []

        if self.pooling_mode in {"max", "max_mean"}:
            pooled_features.append(torch.amax(input=x, dim=-1))
        
        if self.pooling_mode in {"mean", "max_mean"}:
            pooled_features.append(torch.mean(input=x, dim=-1))
        
        if len(pooled_features) == 1:
            y = pooled_features[0]
        else:
            y = torch.cat(tensors=pooled_features, dim=-1)
        
        if self.should_l2_normalize:
            y = F.normalize(input=y, p=2, dim=-1, eps=1e-12)
        
        return y

# Sketching by sparse random projection
class SparseRandomProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, sparsity_threshold: float = 0.9, random_seed: int = 42, is_signed_projection: bool = True):
        super().__init__()

        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("in_dim and out_dim must be positive")

        if sparsity_threshold < 0.0 or sparsity_threshold >= 1.0:
            raise ValueError("sparsity_threshold must be in [0, 1)")

        random_generator = torch.Generator()
        random_generator.manual_seed(random_seed)

        sparse_random_projection_matrix = torch.randn(size=(out_dim, in_dim), generator=random_generator)

        if is_signed_projection:
            signs = torch.randint(low=0, high=2, size=(out_dim, in_dim), generator=random_generator, dtype=torch.int64)

            signs = signs.float().mul_(other=2.0).sub_(other=1.0)

            sparse_random_projection_matrix = sparse_random_projection_matrix * signs
        
        if sparsity_threshold > 0.0:
            keeps = torch.rand(size=(out_dim, in_dim), generator=random_generator)

            keeps = keeps > sparsity_threshold

            sparse_random_projection_matrix = sparse_random_projection_matrix * keeps.float()
        
        self.register_buffer(name="sparse_random_projection_matrix", tensor=sparse_random_projection_matrix)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.sparse_random_projection_matrix.t()

# Output embedding sparsification with blockwise winners-take-all (WTA)
class BlockwiseWTA(nn.Module):
    def __init__(self, topk_per_block: int = 8, num_blocks: int = 8, is_binary_sparsification: bool = False):
        super().__init__()

        if topk_per_block <= 0 or num_blocks <= 0:
            raise ValueError("topk_per_block and num_blocks must be positive")
        
        self.topk_per_block = topk_per_block
        self.num_blocks = num_blocks
        self.is_binary_sparsification = is_binary_sparsification
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, embedding_dim = x.shape

        if embedding_dim % self.num_blocks != 0:
            raise ValueError("embedding_dim must be divisible by num_blocks")
        
        embedding_block_size = embedding_dim // self.num_blocks

        sparsified_x = torch.zeros_like(input=x)

        for block_idx in range(self.num_blocks):
            block_starting_position = block_idx * embedding_block_size
            block_ending_position = block_starting_position + embedding_block_size

            embedding_block = x[:, block_starting_position: block_ending_position]

            topk_values, topk_indices = torch.topk(input=embedding_block, k=min(self.topk_per_block, embedding_block_size), dim=1, largest=True, sorted=False)

            if self.is_binary_sparsification:
                sparsified_x.scatter_(dim=1, index=topk_indices + block_starting_position, src=torch.ones_like(input=topk_values))
            else:
                sparsified_x.scatter_(dim=1, index=topk_indices + block_starting_position, src=topk_values)
        
        return sparsified_x

# PotHash instance configuration
@dataclass
class PotHashConfiguration:
    # Input sequence preprocessing
    alphabet: list[str] = field(default_factory=lambda: dna_alphabet.copy())
    should_convert_rna_to_dna: bool = True
    # Optional hard upperbound on sequence length after preprocessing
    max_sequence_length: Optional[int] = None

    # Submer selection
    should_use_submers: bool = True
    submer_selection_kmer_size: int = 15
    submer_selection_window_size: int = 20

    # Sequence embedding
    should_use_sequence_embedding: bool = False
    sequence_embedding_dim: int = 8

    # Convolution block
    convolution_branch_kernel_sizes: Tuple[int, ...] = (3, 5, 9)
    convolution_branch_out_channels: int = 8
    should_use_convolution_branch_batchnorm: bool = True

    # Pooling and normalization
    # Available pooling modes: {"max", "mean", "max_mean"}
    pooling_mode: str = "max_mean"
    should_l2_normalize_after_pooling: bool = True

    # Sparse random projection
    projection_out_dim: int = 512
    projection_sparsity_threshold: float = 0.9
    projection_random_seed: int = 42
    is_signed_projection: bool = True

    # Sparsification with blockwise winners-take-all (WTA)
    wta_topk_per_block: int = 8
    wta_num_blocks: int = 8
    wta_is_binary_sparsification: bool = False

    # Miscellaneous
    miscellaneous_random_seed: int = 42

# Complete PotHash model
class PotHash(nn.Module):
    def __init__(self, config: PotHashConfiguration):
        super().__init__()

        self.config = config

        # Choosing sequence encoder
        if config.should_use_sequence_embedding:
            self.sequence_encoder: nn.Module = LearnedEmbeddingEncoder(alphabet=config.alphabet, embedding_dim=config.sequence_embedding_dim)

            sequence_encoder_out_channels = config.sequence_embedding_dim
        else:
            self.sequence_encoder: nn.Module = OneHotEncoder(alphabet=config.alphabet)

            sequence_encoder_out_channels = len(config.alphabet)
        
        # Setting up submer selector
        self.minimizer_masker = MinimizerMasker(kmer_size=config.submer_selection_kmer_size, window_size=config.submer_selection_window_size)

        # Setting up multi-scale convolution block
        self.multi_scale_convolution_block: nn.Module = MultiScaleConvolutionBlock(in_channels=sequence_encoder_out_channels, out_channels=config.convolution_branch_out_channels, branch_kernel_sizes=config.convolution_branch_kernel_sizes, should_use_batchnorm=config.should_use_convolution_branch_batchnorm)

        # Setting up pooling and normalization block
        pooling_out_dim_multiplier = 2 if config.pooling_mode == "max_mean" else 1
        convolution_block_out_channels = self.multi_scale_convolution_block.out_channels
        pooling_out_features_dim = pooling_out_dim_multiplier * convolution_block_out_channels

        self.pooling_and_normalization_block: nn.Module = PoolingAndNormalization(pooling_mode=config.pooling_mode, should_l2_normalize=config.should_l2_normalize_after_pooling)

        # Setting up sparse random projector
        self.sparse_random_projector: nn.Module = SparseRandomProjection(in_dim=pooling_out_features_dim, out_dim=config.projection_out_dim, sparsity_threshold=config.projection_sparsity_threshold, random_seed=config.projection_random_seed, is_signed_projection=config.is_signed_projection)

        # Setting up blockwise winners-take-all (WTA) sparsifier
        self.blockwise_wta_sparsifier: nn.Module = BlockwiseWTA(topk_per_block=config.wta_topk_per_block, num_blocks=config.wta_num_blocks, is_binary_sparsification=config.wta_is_binary_sparsification)
    
    @staticmethod
    def _truncate_sequence(sequence: str, max_sequence_length: Optional[int] = None) -> str:
        if max_sequence_length is None:
            return sequence
        
        return sequence[: max_sequence_length]
    
    def _encode_sequence(self, sequence: str) -> torch.Tensor:
        device = get_device()

        preprocessed_sequence = self._truncate_sequence(sequence=preprocess_sequence(sequence=sequence, should_convert_rna_to_dna=self.config.should_convert_rna_to_dna), max_sequence_length=self.config.max_sequence_length)

        # Encoding sequence
        # Input: L-length sequence -> output: [1, C, L] dimensional tensor
        # Here, C is number of encoding channels (batch size B is 1 here) and L is sequence length after preprocessing and truncation
        encoded_sequence = self.sequence_encoder(preprocessed_sequence).to(device)

        # Optional masking of sequence encoding by submers selection
        if self.config.should_use_submers:
            # Input: L-length sequence -> output: [L] dimensional tensor of 0's and 1's
            # Here, 1's indicate the positions of submers in the sequence
            encoded_sequence_mask = self.minimizer_masker(sequence=preprocessed_sequence, hash_function=encode_kmer_to_int).to(device)

            if encoded_sequence_mask.numel() > 0:
                # encoded_sequence (new): [1, C, L] = encoded_sequence (old): [1, C, L] * encoded_sequence_mask: [L].view(1, 1, -1) -> [1, 1, L]
                encoded_sequence = encoded_sequence * encoded_sequence_mask.view(1, 1, -1)
            
        return encoded_sequence
    
    def forward(self, sequence: str) -> torch.Tensor:
        # Input: L-length sequence -> output: [1, C, L] dimensional tensor
        # Here, C is number of encoding channels (batch size B is 1 here)
        x = self._encode_sequence(sequence=sequence)

        # Input: [1, C, L] dimensional tensor -> output: [1, C', L] dimensional tensor
        # Here, C' is number of convolution channels after multi-scale convolution concatenation
        y = self.multi_scale_convolution_block(x)

        # Input: [1, C', L] dimensional tensor -> output: [1, D_feat] dimensional tensor
        # Here, D_feat is sequence features dimension after global max/mean pooling along sequence length (L) dimension
        # D_feat is either C' or 2C' (concatenation of max and mean pooled features if done both)
        # D_feat does not depend on the length of the sequence L, ensuring that the sequence features dimension is invariant to sequence length
        y = self.pooling_and_normalization_block(y)

        # Input: [1, D_feat] dimensional tensor -> output: [1, D_proj] dimensional tensor
        # Here, D_proj is projected dimension after sparse random projection
        y = self.sparse_random_projector(y)

        # Input: [1, D_proj] dimensional tensor -> output: [1, D_proj] dimensional tensor
        # Blockwise winners-take-all (WTA) sparsification does not change the projected sequence features dimension D_proj
        # Regardless of the length of the sequence L, the projected sequence features dimension D_proj is always the same
        y = self.blockwise_wta_sparsifier(y)

        return y
    
    @torch.no_grad()
    def compute_sequence_similarity(self, sequence_a: str, sequence_b: str, similarity_measurement_metric: str = "cosine") -> float:
        # This function computes the similarity between sequence_a and sequence_b in the hash space
        hashcode_a = self.forward(sequence=sequence_a)
        hashcode_b = self.forward(sequence=sequence_b)

        if similarity_measurement_metric == "cosine":
            # Sequence similarity measurement with cosine similarity for real-valued hashcodes
            normalized_hashcode_a = F.normalize(hashcode_a, p=2, dim=-1, eps=1e-12)
            normalized_hashcode_b = F.normalize(hashcode_b, p=2, dim=-1, eps=1e-12)

            return float((normalized_hashcode_a * normalized_hashcode_b).sum().item())
        elif similarity_measurement_metric == "hamming":
            # Sequence similarity measurement with Hamming distance for binary hashcodes
            # Binarization of hashcodes just to make sure the hamming distance is computed on binary hashcodes
            binarized_hashcode_a = (hashcode_a != 0).to(torch.uint8)
            binarized_hashcode_b = (hashcode_b != 0).to(torch.uint8)

            return float((binarized_hashcode_a != binarized_hashcode_b).sum().item())
        else:
            raise ValueError("similarity_measurement_metric must be one of {'cosine', 'hamming'}")

# Optional simple contrastive trainer for later downstream sequence similarity learning experiments with PotHash model
class PotHashContrastiveTrainer(nn.Module):
    def __init__(self, base_pothash_model: PotHash, trainable_dim: int = 128):
        super().__init__()

        self.base_pothash_model = base_pothash_model

        self.pothash_contrastive_learner = nn.Sequential(nn.Linear(in_features=base_pothash_model.config.projection_out_dim, out_features=trainable_dim), nn.ReLU(), nn.Linear(in_features=trainable_dim, out_features=trainable_dim))

    def forward(self, sequence: str) -> torch.Tensor:
        x = self.base_pothash_model(sequence)

        y = self.pothash_contrastive_learner(x)

        return y

# User interface functions
def build_pothash_evaluation_model() -> PotHash:
    pothash_model_config = PotHashConfiguration()

    set_seeds_globally_for_reproducibility(seed=pothash_model_config.miscellaneous_random_seed)

    pothash_model = PotHash(config=pothash_model_config).to(device=get_device()).eval()

    return pothash_model

if __name__ == "__main__":
    pothash = build_pothash_evaluation_model()

    sequence_1 = "ACGTACGTACGTACGTACGT"; sequence_2 = "ACGTTCGTAGGTACCTACGA"; sequence_3 = "TGCATGCATGCATGCATGCA"

    hashcode_1 = pothash(sequence_1); hashcode_2 = pothash(sequence_2); hashcode_3 = pothash(sequence_3)

    assert hashcode_1.shape == hashcode_2.shape == hashcode_3.shape

    print(f"Hashcode shape: {hashcode_1.shape}")

    print(f"Cosine similarity between sequence_1 and sequence_2: {pothash.compute_sequence_similarity(sequence_a=sequence_1, sequence_b=sequence_2, similarity_measurement_metric='cosine'):.4f}")
    print(f"Cosine similarity between sequence_2 and sequence_3: {pothash.compute_sequence_similarity(sequence_a=sequence_2, sequence_b=sequence_3, similarity_measurement_metric='cosine'):.4f}")
    print(f"Cosine similarity between sequence_3 and sequence_1: {pothash.compute_sequence_similarity(sequence_a=sequence_3, sequence_b=sequence_1, similarity_measurement_metric='cosine'):.4f}")