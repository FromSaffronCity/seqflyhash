from dataclasses import dataclass, field
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import Optional
import random
from typing import Sequence
import torch
from typing import Tuple
from typing import Union

dna_alphabet = ['A', 'C', 'G', 'T']

# Base codes: A, C, G, T -> 0, 1, 2, 3; unknown base -> 4; padding -> 5
unknown_base_code = len(dna_alphabet)
padding_base_code = len(dna_alphabet) + 1

# Utility functions
def convert_sequence_to_base_codes(sequence: str) -> np.ndarray:
    # A, C, G, T -> 0, 1, 2, 3 and every other character -> 4 (unknown)
    base_code_lookup_table = np.full(shape=256, fill_value=unknown_base_code, dtype=np.int64)

    for base_code, base in enumerate(dna_alphabet):
        base_code_lookup_table[ord(base)] = base_code

    return base_code_lookup_table[np.frombuffer(sequence.encode("ascii", errors="replace"), dtype=np.uint8)]

def compute_reverse_complement_base_codes(base_codes: np.ndarray) -> np.ndarray:
    reverse_complement_base_codes = base_codes[:: -1].copy()

    is_known_base = reverse_complement_base_codes < unknown_base_code

    # A <-> T and C <-> G, since the alphabet order is A, C, G, T
    reverse_complement_base_codes[is_known_base] = 3 - reverse_complement_base_codes[is_known_base]

    return reverse_complement_base_codes

def mix_uint64_hash(values: np.ndarray, seed: int = 0) -> np.ndarray:
    # SplitMix64 finalizer, a bijective and well-mixing hash on 64-bit integers
    with np.errstate(over="ignore"):
        x = values.astype(np.uint64) ^ np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))

    return x

def compute_canonical_kmer_hash_values(base_codes: np.ndarray, kmer_size: int, seed: int = 0) -> np.ndarray:
    # Returns one hash value per k-mer starting position, identical for a k-mer and its reverse complement
    # K-mers containing unknown bases get the maximum hash value so that they are never selected as minimizers
    if kmer_size > 31:
        raise ValueError("kmer_size must be <= 31 to fit 2-bit k-mer codes into 64 bits")

    num_kmers = len(base_codes) - kmer_size + 1

    if num_kmers <= 0:
        return np.zeros(shape=(0,), dtype=np.uint64)

    kmer_windows = np.lib.stride_tricks.sliding_window_view(base_codes, window_shape=kmer_size)

    is_invalid_kmer = (kmer_windows >= unknown_base_code).any(axis=1)

    clipped_kmer_windows = np.minimum(kmer_windows, 3).astype(np.uint64)

    powers_of_four = np.uint64(4) ** np.arange(kmer_size - 1, -1, -1, dtype=np.uint64)

    forward_kmer_codes = clipped_kmer_windows @ powers_of_four
    reverse_complement_kmer_codes = (np.uint64(3) - clipped_kmer_windows) @ powers_of_four[:: -1]

    canonical_kmer_hash_values = mix_uint64_hash(values=np.minimum(forward_kmer_codes, reverse_complement_kmer_codes), seed=seed)

    canonical_kmer_hash_values[is_invalid_kmer] = np.iinfo(np.uint64).max

    return canonical_kmer_hash_values

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
# Both encoders take a [B, L] tensor of base codes and return a [B, C, L] tensor, where padding positions are all-zero columns
class OneHotEncoder(nn.Module):
    def __init__(self, alphabet: list = dna_alphabet):
        super().__init__()

        self.alphabet = alphabet
        self.num_channels = len(alphabet)
    
    def forward(self, base_codes: torch.Tensor) -> torch.Tensor:
        # Unknown bases and padding become all-zero columns
        one_hot_encoding = F.one_hot(base_codes, num_classes=self.num_channels + 2)[..., : self.num_channels]

        return one_hot_encoding.permute(0, 2, 1).float()

class LearnedEmbeddingEncoder(nn.Module):
    def __init__(self, alphabet: list = dna_alphabet, embedding_dim: int = 8):
        super().__init__()

        self.alphabet = alphabet

        # One additional embedding for unknown bases (learned) and one for padding (fixed at zero)
        self.embedding_table = nn.Embedding(num_embeddings=len(alphabet) + 2, embedding_dim=embedding_dim, padding_idx=len(alphabet) + 1)
    
    def forward(self, base_codes: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding_table(base_codes)

        return embedding.permute(0, 2, 1)

# Submer selection
class MinimizerMasker:
    def __init__(self, kmer_size: int = 9, window_size: int = 14, hash_seed: int = 42):
        if kmer_size <= 0 or window_size <= 0:
            raise ValueError("kmer_size and window_size must be > 0")
        
        if kmer_size > window_size:
            raise ValueError("kmer_size must be <= window_size")

        if kmer_size % 2 == 0:
            raise ValueError("kmer_size must be odd so that submer centers map exactly onto the reverse complement strand")
        
        self.kmer_size = kmer_size
        self.window_size = window_size
        self.hash_seed = hash_seed

    def select_minimizer_starting_positions(self, sequence: str) -> np.ndarray:
        # Canonical minimizers with a random hash: the same k-mers are selected on both strands and selection is not biased towards A-rich k-mers
        # Returns a boolean array over k-mer starting positions, where ties within a window select every tied position
        kmer_hash_values = compute_canonical_kmer_hash_values(base_codes=convert_sequence_to_base_codes(sequence=sequence), kmer_size=self.kmer_size, seed=self.hash_seed)

        num_kmers_per_window = self.window_size - self.kmer_size + 1

        is_minimizer = np.zeros(shape=(len(kmer_hash_values),), dtype=bool)

        if len(kmer_hash_values) < num_kmers_per_window:
            return is_minimizer

        kmer_hash_value_windows = np.lib.stride_tricks.sliding_window_view(kmer_hash_values, window_shape=num_kmers_per_window)

        window_minimum_hash_values = kmer_hash_value_windows.min(axis=1)

        is_window_minimum = (kmer_hash_value_windows == window_minimum_hash_values[:, None]) & (window_minimum_hash_values[:, None] != np.iinfo(np.uint64).max)

        window_indices, kmer_offsets_in_window = np.nonzero(is_window_minimum)

        is_minimizer[window_indices + kmer_offsets_in_window] = True

        return is_minimizer

    def __call__(self, sequence: str) -> np.ndarray:
        # Returns a boolean array over sequence positions marking the centers of the selected submers
        submer_center_mask = np.zeros(shape=(len(sequence),), dtype=bool)

        submer_center_mask[np.nonzero(self.select_minimizer_starting_positions(sequence=sequence))[0] + self.kmer_size // 2] = True

        return submer_center_mask

# Multi-scale convolution block
class MultiScaleConvolutionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 8, branch_kernel_sizes: Sequence[int] = (3, 5, 9), should_use_batchnorm: bool = False):
        super().__init__()

        if len(branch_kernel_sizes) == 0:
            raise ValueError("branch_kernel_sizes must be non-empty")

        if any(kernel_size % 2 == 0 for kernel_size in branch_kernel_sizes):
            raise ValueError("branch_kernel_sizes must be odd so that every position feature is centered on its position")
        
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

# Position feature standardization
class PositionFeatureStandardizer(nn.Module):
    # Without centering, a few projection units win the per-position WTA for almost every position of every sequence, which makes unrelated sequences look similar
    def __init__(self, num_features: int):
        super().__init__()

        self.register_buffer(name="feature_means", tensor=torch.zeros(num_features))
        self.register_buffer(name="feature_stds", tensor=torch.ones(num_features))

    @torch.no_grad()
    def calibrate(self, position_features: torch.Tensor) -> None:
        # position_features: [N, C'] features of background positions
        self.feature_means.copy_(position_features.mean(dim=0))
        self.feature_stds.copy_(position_features.std(dim=0).clamp_min(1e-6))

    def forward(self, position_features: torch.Tensor) -> torch.Tensor:
        return (position_features - self.feature_means) / self.feature_stds

# Sketching by sparse random projection
class SparseRandomProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, sparsity_threshold: float = 0.9, random_seed: int = 42, is_signed_projection: bool = True, should_normalize_rows: bool = True):
        super().__init__()

        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("in_dim and out_dim must be positive")

        if sparsity_threshold < 0.0 or sparsity_threshold >= 1.0:
            raise ValueError("sparsity_threshold must be in [0, 1)")

        random_generator = torch.Generator()
        random_generator.manual_seed(random_seed)

        if is_signed_projection:
            # Gaussian weights are already symmetric around zero, so no additional random sign flipping is needed
            sparse_random_projection_matrix = torch.randn(size=(out_dim, in_dim), generator=random_generator)
        else:
            # Unsigned projection, i.e. the binary sampling projection of the original FlyHash
            sparse_random_projection_matrix = torch.ones(size=(out_dim, in_dim))
        
        if sparsity_threshold > 0.0:
            keep_scores = torch.rand(size=(out_dim, in_dim), generator=random_generator)

            keeps = keep_scores > sparsity_threshold

            # Every output unit keeps at least one input connection, otherwise it would be a dead unit that is always zero
            keeps[torch.arange(out_dim), keep_scores.argmax(dim=1)] = True

            sparse_random_projection_matrix = sparse_random_projection_matrix * keeps.float()

        if should_normalize_rows:
            # Unit-norm rows keep units with more or larger connections from winning the WTA disproportionately often
            sparse_random_projection_matrix = F.normalize(sparse_random_projection_matrix, p=2, dim=1, eps=1e-12)
        
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

    def select_winners(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Returns the [N, num_blocks * k] flat indices and values of the winners of every block
        num_rows, embedding_dim = x.shape

        if embedding_dim % self.num_blocks != 0:
            raise ValueError("embedding_dim must be divisible by num_blocks")
        
        embedding_block_size = embedding_dim // self.num_blocks

        topk_values, topk_indices = torch.topk(input=x.view(num_rows, self.num_blocks, embedding_block_size), k=min(self.topk_per_block, embedding_block_size), dim=2, largest=True, sorted=False)

        block_offsets = torch.arange(self.num_blocks, device=x.device).view(1, -1, 1) * embedding_block_size

        winner_indices = (topk_indices + block_offsets).reshape(num_rows, -1)
        winner_values = topk_values.reshape(num_rows, -1)

        if self.is_binary_sparsification:
            winner_values = torch.ones_like(winner_values)

        return winner_indices, winner_values
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        winner_indices, winner_values = self.select_winners(x=x)

        return torch.zeros_like(input=x).scatter_(dim=1, index=winner_indices, src=winner_values)

# PotHash instance configuration
@dataclass
class PotHashConfiguration:
    # Input sequence preprocessing
    alphabet: list[str] = field(default_factory=lambda: dna_alphabet.copy())
    should_convert_rna_to_dna: bool = True
    # Optional hard upperbound on sequence length after preprocessing
    max_sequence_length: Optional[int] = None

    # Submer selection: only positions at the centers of canonical minimizers are hashed, which is faster and keeps the same positions on both strands
    should_use_submers: bool = False
    submer_selection_kmer_size: int = 9
    submer_selection_window_size: int = 14
    submer_selection_hash_seed: int = 42

    # Sequence embedding
    should_use_sequence_embedding: bool = False
    sequence_embedding_dim: int = 8

    # Convolution block
    convolution_branch_kernel_sizes: Tuple[int, ...] = (3, 5, 9)
    convolution_branch_out_channels: int = 16
    should_use_convolution_branch_batchnorm: bool = False

    # Per-position sparse random projection (FlyHash expansion of every position feature)
    projection_out_dim: int = 2048
    projection_sparsity_threshold: float = 0.9
    projection_random_seed: int = 42
    is_signed_projection: bool = True
    should_normalize_projection_rows: bool = True

    # Per-position winners-take-all (WTA): every position votes for its top-k projection units
    position_wta_topk: int = 4
    position_wta_is_binary: bool = False
    # Softmax temperature replacing the hard per-position WTA during training
    position_wta_training_temperature: float = 0.1

    # Pooling of per-position codes over positions and both strands
    # Available pooling modes: {"mean", "max"}
    pooling_mode: str = "mean"

    # Final sparsification with blockwise winners-take-all (WTA)
    wta_topk_per_block: int = 8
    wta_num_blocks: int = 16
    wta_is_binary_sparsification: bool = False

    # Miscellaneous
    miscellaneous_random_seed: int = 42
    # Number of positions projected at once, which bounds memory usage for long sequences
    position_chunk_size: int = 16384

# Complete PotHash model
# Sequence -> both strands -> encoding -> multi-scale convolution -> per-position feature standardization -> per-position sparse random projection and WTA
# -> pooling over positions and both strands -> blockwise WTA -> sparse hashcode that is invariant to sequence length and reverse complementation
class PotHash(nn.Module):
    def __init__(self, config: PotHashConfiguration):
        super().__init__()

        if config.pooling_mode not in {"mean", "max"}:
            raise ValueError("pooling_mode must be one of {'mean', 'max'}")

        self.config = config

        # Choosing sequence encoder
        if config.should_use_sequence_embedding:
            self.sequence_encoder: nn.Module = LearnedEmbeddingEncoder(alphabet=config.alphabet, embedding_dim=config.sequence_embedding_dim)

            sequence_encoder_out_channels = config.sequence_embedding_dim
        else:
            self.sequence_encoder: nn.Module = OneHotEncoder(alphabet=config.alphabet)

            sequence_encoder_out_channels = len(config.alphabet)
        
        # Setting up submer selector
        self.minimizer_masker = MinimizerMasker(kmer_size=config.submer_selection_kmer_size, window_size=config.submer_selection_window_size, hash_seed=config.submer_selection_hash_seed)

        # Setting up multi-scale convolution block
        self.multi_scale_convolution_block: nn.Module = MultiScaleConvolutionBlock(in_channels=sequence_encoder_out_channels, out_channels=config.convolution_branch_out_channels, branch_kernel_sizes=config.convolution_branch_kernel_sizes, should_use_batchnorm=config.should_use_convolution_branch_batchnorm)

        # Setting up per-position feature standardizer
        self.position_feature_standardizer = PositionFeatureStandardizer(num_features=self.multi_scale_convolution_block.out_channels)

        # Setting up per-position sparse random projector
        self.sparse_random_projector: nn.Module = SparseRandomProjection(in_dim=self.multi_scale_convolution_block.out_channels, out_dim=config.projection_out_dim, sparsity_threshold=config.projection_sparsity_threshold, random_seed=config.projection_random_seed, is_signed_projection=config.is_signed_projection, should_normalize_rows=config.should_normalize_projection_rows)

        # Setting up per-position and final blockwise winners-take-all (WTA) sparsifiers
        self.position_wta_sparsifier = BlockwiseWTA(topk_per_block=config.position_wta_topk, num_blocks=1, is_binary_sparsification=config.position_wta_is_binary)
        self.blockwise_wta_sparsifier: nn.Module = BlockwiseWTA(topk_per_block=config.wta_topk_per_block, num_blocks=config.wta_num_blocks, is_binary_sparsification=config.wta_is_binary_sparsification)

        # Expected pooled code of background sequences, subtracted before the final WTA so that only units over-represented in a sequence win
        self.register_buffer(name="background_pooled_code", tensor=torch.zeros(config.projection_out_dim))

    @property
    def device(self) -> torch.device:
        return self.sparse_random_projector.sparse_random_projection_matrix.device
    
    @staticmethod
    def _truncate_sequence(sequence: str, max_sequence_length: Optional[int] = None) -> str:
        if max_sequence_length is None:
            return sequence
        
        return sequence[: max_sequence_length]

    def _prepare_batch(self, sequences: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Returns [2B, L_max] base codes and [2B, L_max] boolean masks of hashed positions
        # Rows 0..B-1 are the forward strands and rows B..2B-1 are the reverse complement strands
        preprocessed_sequences = [self._truncate_sequence(sequence=preprocess_sequence(sequence=sequence, should_convert_rna_to_dna=self.config.should_convert_rna_to_dna), max_sequence_length=self.config.max_sequence_length) for sequence in sequences]

        max_sequence_length = max(1, max(len(sequence) for sequence in preprocessed_sequences))

        num_sequences = len(preprocessed_sequences)

        base_codes = np.full(shape=(2 * num_sequences, max_sequence_length), fill_value=padding_base_code, dtype=np.int64)
        hashed_position_masks = np.zeros(shape=(2 * num_sequences, max_sequence_length), dtype=bool)

        for sequence_idx, sequence in enumerate(preprocessed_sequences):
            sequence_length = len(sequence)

            if sequence_length == 0:
                continue

            forward_base_codes = convert_sequence_to_base_codes(sequence=sequence)

            base_codes[sequence_idx, : sequence_length] = forward_base_codes
            base_codes[num_sequences + sequence_idx, : sequence_length] = compute_reverse_complement_base_codes(base_codes=forward_base_codes)

            hashed_position_mask = self.minimizer_masker(sequence=sequence) if self.config.should_use_submers else np.ones(shape=(sequence_length,), dtype=bool)

            if not hashed_position_mask.any():
                # No submer could be selected (sequence shorter than the window or too many unknown bases), so every position is hashed
                hashed_position_mask = np.ones(shape=(sequence_length,), dtype=bool)

            hashed_position_masks[sequence_idx, : sequence_length] = hashed_position_mask
            hashed_position_masks[num_sequences + sequence_idx, : sequence_length] = hashed_position_mask[:: -1]

        return torch.from_numpy(base_codes).to(self.device), torch.from_numpy(hashed_position_masks).to(self.device)

    def compute_position_features(self, sequences: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Returns [N, C'] standardized features of all hashed positions and the [N] row indices (into the 2B strands) they belong to
        base_codes, hashed_position_masks = self._prepare_batch(sequences=sequences)

        # [2B, L] -> [2B, C, L] -> [2B, C', L] -> [2B, L, C']
        position_features = self.multi_scale_convolution_block(self.sequence_encoder(base_codes)).permute(0, 2, 1)

        strand_indices, position_indices = torch.nonzero(hashed_position_masks, as_tuple=True)

        return self.position_feature_standardizer(position_features[strand_indices, position_indices]), strand_indices

    def _pool_position_codes(self, pooled_codes: torch.Tensor, position_count_per_strand: torch.Tensor) -> torch.Tensor:
        # [2B, D] strand codes -> [B, D] sequence codes, combined symmetrically over both strands for reverse complement invariance
        num_sequences = pooled_codes.shape[0] // 2

        if self.config.pooling_mode == "mean":
            pooled_codes = pooled_codes / position_count_per_strand.clamp_min(1).unsqueeze(1)

            return 0.5 * (pooled_codes[: num_sequences] + pooled_codes[num_sequences:])

        return torch.maximum(pooled_codes[: num_sequences], pooled_codes[num_sequences:])

    def compute_pooled_codes(self, sequences: Union[str, Sequence[str]], training_temperature: Optional[float] = None) -> torch.Tensor:
        # Returns [B, D] dense pooled codes before the final WTA
        # With training_temperature, the hard per-position WTA is replaced by a differentiable softmax over projection units
        if isinstance(sequences, str):
            sequences = [sequences]

        position_features, strand_indices = self.compute_position_features(sequences=sequences)

        num_strands = 2 * len(sequences)

        position_count_per_strand = torch.bincount(strand_indices, minlength=num_strands).float()

        if training_temperature is not None:
            if self.config.pooling_mode != "mean":
                raise ValueError("training requires pooling_mode='mean'")

            position_codes = torch.softmax(self.sparse_random_projector(position_features) / training_temperature, dim=1)

            pooled_codes = torch.zeros(size=(num_strands, self.config.projection_out_dim), device=self.device, dtype=position_codes.dtype).index_add(0, strand_indices, position_codes)
        else:
            pooled_codes = torch.zeros(size=(num_strands, self.config.projection_out_dim), device=self.device)

            for chunk_start in range(0, position_features.shape[0], self.config.position_chunk_size):
                chunk_end = chunk_start + self.config.position_chunk_size

                winner_indices, winner_values = self.position_wta_sparsifier.select_winners(x=self.sparse_random_projector(position_features[chunk_start: chunk_end]))

                chunk_strand_indices = strand_indices[chunk_start: chunk_end].unsqueeze(1).expand_as(winner_indices)

                if self.config.pooling_mode == "mean":
                    pooled_codes.index_put_((chunk_strand_indices.reshape(-1), winner_indices.reshape(-1)), winner_values.reshape(-1), accumulate=True)
                else:
                    pooled_codes[chunk_strand_indices.reshape(-1), winner_indices.reshape(-1)] = torch.maximum(pooled_codes[chunk_strand_indices.reshape(-1), winner_indices.reshape(-1)], winner_values.reshape(-1))

        return self._pool_position_codes(pooled_codes=pooled_codes, position_count_per_strand=position_count_per_strand)
    
    def forward(self, sequences: Union[str, Sequence[str]]) -> torch.Tensor:
        # Input: a sequence or a list of B sequences of any lengths -> output: [B, D] sparse hashcodes
        # D does not depend on sequence lengths, and a sequence and its reverse complement get the same hashcode
        pooled_codes = F.normalize(self.compute_pooled_codes(sequences=sequences) - self.background_pooled_code, p=2, dim=1, eps=1e-12)

        return self.blockwise_wta_sparsifier(pooled_codes)

    @torch.no_grad()
    def hash_sequences(self, sequences: Sequence[str], batch_size: int = 32) -> torch.Tensor:
        # Hashes many sequences in batches, sorted by length to minimize padding
        sorted_sequence_indices = sorted(range(len(sequences)), key=lambda sequence_idx: len(sequences[sequence_idx]))

        hashcodes = torch.zeros(size=(len(sequences), self.config.projection_out_dim), device=self.device)

        for batch_start in range(0, len(sequences), batch_size):
            batch_sequence_indices = sorted_sequence_indices[batch_start: batch_start + batch_size]

            hashcodes[batch_sequence_indices] = self.forward(sequences=[sequences[sequence_idx] for sequence_idx in batch_sequence_indices])

        return hashcodes

    @torch.no_grad()
    def calibrate_background_statistics(self, sequences: Sequence[str]) -> None:
        # Estimates per-feature means and standard deviations of hashed positions and the expected pooled code on background sequences
        self.position_feature_standardizer.feature_means.zero_()
        self.position_feature_standardizer.feature_stds.fill_(1.0)

        position_features, _ = self.compute_position_features(sequences=sequences)

        self.position_feature_standardizer.calibrate(position_features=position_features)

        self.background_pooled_code.copy_(self.compute_pooled_codes(sequences=sequences).mean(dim=0))
    
    @staticmethod
    def compute_hashcode_similarity(hashcode_a: torch.Tensor, hashcode_b: torch.Tensor, similarity_measurement_metric: str = "cosine") -> torch.Tensor:
        # Row-wise similarity between [B, D] hashcodes
        if similarity_measurement_metric == "cosine":
            # Sequence similarity measurement with cosine similarity for real-valued hashcodes
            return (F.normalize(hashcode_a, p=2, dim=-1, eps=1e-12) * F.normalize(hashcode_b, p=2, dim=-1, eps=1e-12)).sum(dim=-1)
        elif similarity_measurement_metric == "hamming":
            # Sequence similarity measurement with fractional Hamming similarity (1 - normalized Hamming distance) for binary hashcodes
            # Binarization of hashcodes just to make sure the hamming distance is computed on binary hashcodes
            return 1.0 - ((hashcode_a != 0) != (hashcode_b != 0)).float().mean(dim=-1)
        elif similarity_measurement_metric == "jaccard":
            # Jaccard similarity between the sets of active hashcode units
            binarized_hashcode_a = hashcode_a != 0
            binarized_hashcode_b = hashcode_b != 0

            return (binarized_hashcode_a & binarized_hashcode_b).sum(dim=-1).float() / (binarized_hashcode_a | binarized_hashcode_b).sum(dim=-1).clamp_min(1).float()
        else:
            raise ValueError("similarity_measurement_metric must be one of {'cosine', 'hamming', 'jaccard'}")

    @torch.no_grad()
    def compute_sequence_similarity(self, sequence_a: str, sequence_b: str, similarity_measurement_metric: str = "cosine") -> float:
        # This function computes the similarity between sequence_a and sequence_b in the hash space
        hashcodes = self.forward(sequences=[sequence_a, sequence_b])

        return float(self.compute_hashcode_similarity(hashcode_a=hashcodes[0: 1], hashcode_b=hashcodes[1: 2], similarity_measurement_metric=similarity_measurement_metric).item())

# User interface functions
def build_pothash_evaluation_model(config: Optional[PotHashConfiguration] = None, calibration_sequences: Optional[Sequence[str]] = None) -> PotHash:
    pothash_model_config = config if config is not None else PotHashConfiguration()

    set_seeds_globally_for_reproducibility(seed=pothash_model_config.miscellaneous_random_seed)

    pothash_model = PotHash(config=pothash_model_config).to(device=get_device()).eval()

    if calibration_sequences is None:
        # Default background: random sequences with uniform base composition, generated with a fixed seed
        calibration_random_generator = np.random.default_rng(seed=pothash_model_config.miscellaneous_random_seed)

        calibration_sequences = [''.join(calibration_random_generator.choice(dna_alphabet, size=1000)) for _ in range(20)]

    pothash_model.calibrate_background_statistics(sequences=calibration_sequences)

    return pothash_model

if __name__ == "__main__":
    pothash = build_pothash_evaluation_model()

    sequence_1 = "ACGTACGTACGTACGTACGT"; sequence_2 = "ACGTTCGTAGGTACCTACGA"; sequence_3 = "TGCATGCATGCATGCATGCA"

    hashcodes = pothash(sequences=[sequence_1, sequence_2, sequence_3])

    print(f"Hashcode shape: {hashcodes.shape}")

    print(f"Cosine similarity between sequence_1 and sequence_2: {pothash.compute_sequence_similarity(sequence_a=sequence_1, sequence_b=sequence_2, similarity_measurement_metric='cosine'):.4f}")
    print(f"Cosine similarity between sequence_2 and sequence_3: {pothash.compute_sequence_similarity(sequence_a=sequence_2, sequence_b=sequence_3, similarity_measurement_metric='cosine'):.4f}")
    print(f"Cosine similarity between sequence_3 and sequence_1: {pothash.compute_sequence_similarity(sequence_a=sequence_3, sequence_b=sequence_1, similarity_measurement_metric='cosine'):.4f}")
