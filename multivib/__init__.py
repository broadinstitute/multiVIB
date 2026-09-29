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
from .checkpoint import save_checkpoint, load_checkpoint, setup_model_from_checkpoint, migrate_checkpoint
from .preprocessing import normalize_log1p, preprocess_h5ad
from .mapping import (
    match_features,
    get_embedding,
    map_query_to_reference,
    map_species_query_to_reference,
)
from .training import (
    multivib_vertical_training,
    multivib_horizontal_training,
    multivib_species_training,
    multivibR_training,
)
from .utils import (
    crossover_augmentation,
    init_weights,
    one_hot,
    scale_by_batch,
    kl_annealing_weight,
    EMA,
)

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
    # checkpoint
    "save_checkpoint",
    "load_checkpoint",
    "setup_model_from_checkpoint",
    "migrate_checkpoint",
    # preprocessing
    "normalize_log1p",
    "preprocess_h5ad",
    # mapping
    "match_features",
    "get_embedding",
    "map_query_to_reference",
    "map_species_query_to_reference",
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
    "kl_annealing_weight",
    "EMA",
]
