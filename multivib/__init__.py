"""
multiVIB: A Unified Probabilistic Contrastive Learning Framework
for Atlas-Scale Integration of Single-Cell Multi-Omics Data.
"""

from .layers import MaskedLinear, LoRALinear, VariationalEncoder, CellTypeClassifier
from .losses import (
    DCL,
    OnlinePrototypeClustering,
    SinkhornOTLoss,
    OODAlignmentLoss,
    GraphNeighborhoodReg,
    VICRegLoss,
)
from .models import multivib, multivibLoRA, multivibS, multivibLoRAS, multivibR
from .training import (
    multivib_vertical_training,
    multivib_horizontal_training,
    multivib_species_training,
    multivibR_training,
)
from .utils import crossover_augmentation, init_weights, one_hot, scale_by_batch

__version__ = "0.1.0"

__all__ = [
    # layers
    "MaskedLinear",
    "LoRALinear",
    "VariationalEncoder",
    "CellTypeClassifier",
    # losses
    "DCL",
    "OnlinePrototypeClustering",
    "SinkhornOTLoss",
    "OODAlignmentLoss",
    "GraphNeighborhoodReg",
    "VICRegLoss",
    # models
    "multivib",
    "multivibLoRA",
    "multivibS",
    "multivibLoRAS",
    "multivibR",
    # training
    "multivib_vertical_training",
    "multivib_horizontal_training",
    "multivib_species_training",
    "multivibR_training",
    # utils
    "crossover_augmentation",
    "init_weights",
    "one_hot",
    "scale_by_batch",
]
