"""SCH-KANH mainline forward-only pipeline (pre-training stage)."""

from .config import SchKanhConfig, load_sch_kahn_config
from .loss_input_builder import LossBatchInputs, build_loss_batch_inputs
from .losses import loss_bal, loss_grl, loss_q, loss_sem, pairwise_cosine_rows
from .mainline_forward import MainlineForwardOutput, SchKanhStats, run_sch_kahn_mainline
from .semantic_supervision_reader import (
    SemanticSupervision,
    SemanticSupervisionMatrix,
    load_semantic_supervision,
    resolve_semantic_cache_dir,
)

__all__ = [
    "SchKanhConfig",
    "SemanticSupervision",
    "SemanticSupervisionMatrix",
    "MainlineForwardOutput",
    "SchKanhStats",
    "LossBatchInputs",
    "build_loss_batch_inputs",
    "pairwise_cosine_rows",
    "loss_sem",
    "loss_q",
    "loss_bal",
    "loss_grl",
    "load_semantic_supervision",
    "resolve_semantic_cache_dir",
    "load_sch_kahn_config",
    "run_sch_kahn_mainline",
]
