"""Bounded class-balanced spatial-feature memory with optional FAISS PQ."""

from dataclasses import dataclass, field

import faiss
import numpy as np
import torch

LatentExemplar = tuple[np.ndarray, torch.Tensor]


@dataclass
class LatentMemory:
    """Store spatial features in deterministic per-label reservoirs."""

    num_exemplars_per_class: int = 100
    seed: int = 42
    use_quantization: bool = False
    num_groups: int = 4
    nbits: int = 8
    reservoirs: dict[int, list[LatentExemplar]] = field(default_factory=dict, init=False, repr=False)
    seen_counts: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    pq: faiss.ProductQuantizer | None = field(default=None, init=False, repr=False)
    spatial_shape: tuple[int, int, int] | None = field(default=None, init=False)
    _generator: torch.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize deterministic sampling state."""
        self._generator = torch.Generator().manual_seed(int(self.seed))

    def __len__(self) -> int:
        """Return the total number of class-reservoir entries."""
        return sum(len(bucket) for bucket in self.reservoirs.values())

    def fit_quantizer(self, features: torch.Tensor) -> None:
        """Fit product quantization to a batch of NCHW spatial features."""
        if not self.use_quantization:
            return
        _, channels, height, width = features.shape
        if channels % int(self.num_groups) != 0:
            msg = f"Spatial channels ({channels}) must be divisible by num_groups ({self.num_groups})."
            raise ValueError(msg)
        vectors = features.detach().cpu().permute(0, 2, 3, 1).reshape(-1, channels).numpy().astype(np.float32)
        minimum_vectors = 2 ** int(self.nbits)
        if len(vectors) < minimum_vectors:
            msg = f"Product quantization with nbits={self.nbits} requires at least {minimum_vectors} vectors."
            raise ValueError(msg)
        self.pq = faiss.ProductQuantizer(channels, int(self.num_groups), int(self.nbits))
        self.pq.train(vectors)
        self.spatial_shape = (channels, height, width)

    def update(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        """Update per-label reservoirs from NCHW spatial features."""
        if self.spatial_shape is None:
            self.spatial_shape = tuple(int(value) for value in features.shape[1:])
        for feature, target in zip(features.detach().cpu(), targets.detach().cpu(), strict=True):
            stored = self._encode(feature)
            for class_index in torch.nonzero(target > 0, as_tuple=False).flatten().tolist():
                self._update_class(int(class_index), stored, target)

    def sample(self, count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a class-balanced latent batch."""
        classes = sorted(class_index for class_index, bucket in self.reservoirs.items() if bucket)
        if not classes:
            msg = "Cannot sample from an empty latent memory."
            raise ValueError(msg)
        features: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for offset in range(int(count)):
            bucket = self.reservoirs[classes[offset % len(classes)]]
            index = int(torch.randint(len(bucket), (1,), generator=self._generator).item())
            stored, target = bucket[index]
            features.append(self._decode(stored))
            targets.append(target)
        return torch.stack(features).to(device), torch.stack(targets).to(device)

    def _update_class(self, class_index: int, stored: np.ndarray, target: torch.Tensor) -> None:
        capacity = int(self.num_exemplars_per_class)
        if capacity <= 0:
            return
        seen = self.seen_counts.get(class_index, 0) + 1
        self.seen_counts[class_index] = seen
        bucket = self.reservoirs.setdefault(class_index, [])
        exemplar = (stored.copy(), target.clone())
        if len(bucket) < capacity:
            bucket.append(exemplar)
            return
        index = int(torch.randint(seen, (1,), generator=self._generator).item())
        if index < capacity:
            bucket[index] = exemplar

    def _encode(self, feature: torch.Tensor) -> np.ndarray:
        array = feature.numpy().astype(np.float32)
        if not self.use_quantization:
            return array
        if self.pq is None:
            msg = "Fit the product quantizer before adding features."
            raise RuntimeError(msg)
        channels, height, width = array.shape
        vectors = array.transpose(1, 2, 0).reshape(-1, channels)
        return self.pq.compute_codes(vectors).reshape(height, width, -1)

    def _decode(self, stored: np.ndarray) -> torch.Tensor:
        if not self.use_quantization:
            return torch.from_numpy(stored.astype(np.float32, copy=False))
        if self.pq is None or self.spatial_shape is None:
            msg = "Quantized features require a fitted product quantizer."
            raise RuntimeError(msg)
        channels, height, width = self.spatial_shape
        vectors = self.pq.decode(stored.reshape(height * width, -1))
        return torch.from_numpy(vectors.reshape(height, width, channels).transpose(2, 0, 1).copy())
