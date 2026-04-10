from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy import sparse
from torch.utils.data import DataLoader, Subset

from .loss_input_builder import build_s_final_batch
from .semantic_supervision_reader import SemanticSupervision, load_semantic_supervision
from .torch_losses import GrlDomainHead, WeightedLossTerms, loss_bal, loss_grl, loss_q, loss_sem
from .torch_mainline import SchKanhTrainingMainline, TorchMainlineOutput, TrainerSmokeShadowMainline
from .trainer_config import TrainerSmokeConfig, TrainingMainlineConfig
from .trainer_dataset import FeatureCacheTrainerDataset


@dataclass
class TrainerSmokeContext:
    config: TrainerSmokeConfig
    dataset: FeatureCacheTrainerDataset
    dataloader: DataLoader[dict[str, torch.Tensor]]
    supervision: SemanticSupervision
    model: TrainerSmokeShadowMainline
    domain_head: GrlDomainHead
    optimizer: torch.optim.Optimizer
    device: torch.device


@dataclass
class TrainingMainlineContext:
    config: TrainingMainlineConfig
    dataset: FeatureCacheTrainerDataset
    dataloader: DataLoader[dict[str, torch.Tensor]]
    supervision: SemanticSupervision
    model: SchKanhTrainingMainline
    domain_head: GrlDomainHead
    optimizer: torch.optim.Optimizer
    device: torch.device


@dataclass(frozen=True)
class TrainerSmokeStepResult:
    dataset: str
    feature_set_id: str
    semantic_set_id: str
    batch_rows: int
    sample_index_hash: str
    sample_indices: list[int]
    losses: dict[str, float]
    grad_norms: dict[str, float]
    param_delta_norms: dict[str, float]
    checks: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "feature_set_id": self.feature_set_id,
            "semantic_set_id": self.semantic_set_id,
            "batch_rows": self.batch_rows,
            "sample_index_hash": self.sample_index_hash,
            "sample_indices": self.sample_indices,
            "losses": self.losses,
            "grad_norms": self.grad_norms,
            "param_delta_norms": self.param_delta_norms,
            "checks": self.checks,
        }


@dataclass(frozen=True)
class TrainingMainlineStepMetrics:
    epoch: int
    global_step: int
    loss_sem: float
    loss_q: float
    loss_bal: float
    loss_grl: float
    total_loss: float
    lr: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "loss_sem": self.loss_sem,
            "loss_q": self.loss_q,
            "loss_bal": self.loss_bal,
            "loss_grl": self.loss_grl,
            "total_loss": self.total_loss,
            "lr": self.lr,
        }


@dataclass(frozen=True)
class TrainingMainlineRunResult:
    run_dir: str
    metrics_path: str
    checkpoint_path: str
    dataset: str
    feature_set_id: str
    semantic_set_id: str
    sample_index_hash: str
    resumed_from_checkpoint: bool
    start_epoch: int
    completed_epoch: int
    global_step: int
    target_num_epochs: int
    stop_after_epochs: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "metrics_path": self.metrics_path,
            "checkpoint_path": self.checkpoint_path,
            "dataset": self.dataset,
            "feature_set_id": self.feature_set_id,
            "semantic_set_id": self.semantic_set_id,
            "sample_index_hash": self.sample_index_hash,
            "resumed_from_checkpoint": self.resumed_from_checkpoint,
            "start_epoch": self.start_epoch,
            "completed_epoch": self.completed_epoch,
            "global_step": self.global_step,
            "target_num_epochs": self.target_num_epochs,
            "stop_after_epochs": self.stop_after_epochs,
        }


def _resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested torch device `{device_name}` but CUDA is not available")
    return torch.device(device_name)


def _build_optimizer(cfg: Any, context_params: list[torch.nn.Parameter]) -> torch.optim.Optimizer:
    if cfg.optimizer.name == "sgd":
        return torch.optim.SGD(context_params, lr=cfg.optimizer.lr)
    raise RuntimeError(f"Unsupported optimizer: {cfg.optimizer.name}")


def _iter_params(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [param for param in module.parameters() if param.requires_grad]


def _module_grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        total += float(torch.sum(param.grad.detach() ** 2).item())
    return total ** 0.5


def _module_param_snapshot(module: torch.nn.Module) -> list[torch.Tensor]:
    return [param.detach().clone() for param in module.parameters()]


def _module_param_delta_norm(module: torch.nn.Module, before: list[torch.Tensor]) -> float:
    total = 0.0
    for param, old in zip(module.parameters(), before):
        total += float(torch.sum((param.detach() - old) ** 2).item())
    return total ** 0.5


def _fetch_single_batch(context: TrainerSmokeContext) -> dict[str, torch.Tensor]:
    return next(iter(context.dataloader))


def build_trainer_smoke_context(cfg: TrainerSmokeConfig) -> TrainerSmokeContext:
    torch.manual_seed(cfg.runtime.seed)
    device = _resolve_device(cfg.runtime.device)

    feature_cache_dir = cfg.resolve_feature_cache_dir(
        dataset=cfg.smoke.dataset,
        feature_set_id=cfg.smoke.feature_set_id,
    )
    dataset = FeatureCacheTrainerDataset(feature_cache_dir)
    subset = Subset(dataset, cfg.smoke.sample_indices)
    dataloader = DataLoader(subset, batch_size=cfg.smoke.batch_size, shuffle=False, drop_last=False)

    semantic_cache_dir = cfg.resolve_semantic_cache_dir(
        dataset=cfg.smoke.dataset,
        semantic_set_id=cfg.smoke.semantic_set_id,
    )
    supervision = load_semantic_supervision(
        semantic_cache_dir,
        expected_sample_index_hash=dataset.meta.sample_index_hash,
        expected_rows=dataset.meta.rows,
        expected_feature_set_id=dataset.meta.feature_set_id,
    )

    model = TrainerSmokeShadowMainline(
        d_in=dataset.meta.dim,
        d_model=cfg.shadow_mainline.d_model,
        order_k=cfg.shadow_mainline.order_k,
        d_hash=cfg.shadow_mainline.d_hash,
        seed=cfg.runtime.seed,
    ).to(device=device)
    domain_head = GrlDomainHead(
        d_in=cfg.shadow_mainline.d_model,
        lambda_grl=cfg.loss_weights.grl_lambda,
        seed=cfg.runtime.seed + 303,
    ).to(device=device)

    optimizer = _build_optimizer(cfg, _iter_params(model) + _iter_params(domain_head))
    return TrainerSmokeContext(
        config=cfg,
        dataset=dataset,
        dataloader=dataloader,
        supervision=supervision,
        model=model,
        domain_head=domain_head,
        optimizer=optimizer,
        device=device,
    )


def build_training_mainline_context(cfg: TrainingMainlineConfig) -> TrainingMainlineContext:
    torch.manual_seed(cfg.runtime.seed)
    device = _resolve_device(cfg.runtime.device)

    feature_cache_dir = cfg.resolve_feature_cache_dir(
        dataset=cfg.train.dataset,
        feature_set_id=cfg.train.feature_set_id,
    )
    dataset = FeatureCacheTrainerDataset(feature_cache_dir)
    dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, drop_last=False)

    semantic_cache_dir = cfg.resolve_semantic_cache_dir(
        dataset=cfg.train.dataset,
        semantic_set_id=cfg.train.semantic_set_id,
    )
    supervision = load_semantic_supervision(
        semantic_cache_dir,
        expected_sample_index_hash=dataset.meta.sample_index_hash,
        expected_rows=dataset.meta.rows,
        expected_feature_set_id=dataset.meta.feature_set_id,
    )

    model = SchKanhTrainingMainline(
        d_in=dataset.meta.dim,
        d_model=cfg.training_mainline.d_model,
        order_k=cfg.training_mainline.order_k,
        d_hash=cfg.training_mainline.d_hash,
        seed=cfg.runtime.seed,
    ).to(device=device)
    domain_head = GrlDomainHead(
        d_in=cfg.training_mainline.d_model,
        lambda_grl=cfg.loss_weights.grl_lambda,
        seed=cfg.runtime.seed + 303,
    ).to(device=device)

    optimizer = _build_optimizer(cfg, _iter_params(model) + _iter_params(domain_head))
    return TrainingMainlineContext(
        config=cfg,
        dataset=dataset,
        dataloader=dataloader,
        supervision=supervision,
        model=model,
        domain_head=domain_head,
        optimizer=optimizer,
        device=device,
    )


def compute_weighted_loss_terms(
    *,
    output: TorchMainlineOutput,
    s_final_batch: sparse.csr_matrix,
    domain_head: GrlDomainHead,
    cfg: Any,
) -> WeightedLossTerms:
    sem_value = loss_sem(output.v_i, output.v_t, s_final_batch)
    q_value = loss_q(output.v_i, output.v_t, output.b_i, output.b_t)
    bal_value = loss_bal(output.v_i, output.v_t)
    grl_value = loss_grl(output.zout_i, output.zout_t, domain_head)
    total = (
        (cfg.loss_weights.sem * sem_value)
        + (cfg.loss_weights.q * q_value)
        + (cfg.loss_weights.bal * bal_value)
        + (cfg.loss_weights.grl * grl_value)
    )
    return WeightedLossTerms(
        loss_sem=sem_value,
        loss_q=q_value,
        loss_bal=bal_value,
        loss_grl=grl_value,
        total_loss=total,
    )


def _extract_sample_indices(sample_index_tensor: torch.Tensor) -> np.ndarray:
    if sample_index_tensor.ndim != 1:
        raise RuntimeError(f"batch.sample_index must be 1D, got shape {tuple(sample_index_tensor.shape)}")
    sample_indices = sample_index_tensor.detach().cpu().numpy().astype(np.int64, copy=False)
    if sample_indices.size <= 0:
        raise RuntimeError("batch.sample_index must be non-empty")
    return sample_indices


def _validate_batch_supervision(
    *,
    batch_sample_index: torch.Tensor,
    sample_indices: np.ndarray,
    s_final_batch: sparse.csr_matrix,
) -> None:
    expected = batch_sample_index.detach().cpu().numpy().astype(np.int64, copy=False)
    if not np.array_equal(expected, sample_indices):
        raise RuntimeError("batch.sample_index was reordered before S_final slicing")
    batch_rows = int(sample_indices.size)
    if not isinstance(s_final_batch, sparse.csr_matrix):
        raise RuntimeError("training mainline requires S_final_batch to remain CSR sparse matrix")
    if s_final_batch.shape != (batch_rows, batch_rows):
        raise RuntimeError(
            f"S_final batch shape mismatch: got {s_final_batch.shape}, expected {(batch_rows, batch_rows)}"
        )
    if s_final_batch.dtype != np.float32:
        raise RuntimeError(f"S_final batch dtype must stay float32, got {s_final_batch.dtype}")


def run_single_trainer_smoke_step(cfg: TrainerSmokeConfig) -> TrainerSmokeStepResult:
    context = build_trainer_smoke_context(cfg)
    batch = _fetch_single_batch(context)
    sample_index_tensor = batch["sample_index"]
    sample_indices = _extract_sample_indices(sample_index_tensor)
    x_i = batch["x_i"].to(device=context.device, dtype=torch.float32)
    x_t = batch["x_t"].to(device=context.device, dtype=torch.float32)

    before_model = _module_param_snapshot(context.model)
    before_grl = _module_param_snapshot(context.domain_head)

    context.optimizer.zero_grad(set_to_none=True)
    output = context.model(x_i, x_t)
    s_final_batch = build_s_final_batch(sample_indices=sample_indices, supervision=context.supervision)
    _validate_batch_supervision(
        batch_sample_index=sample_index_tensor,
        sample_indices=sample_indices,
        s_final_batch=s_final_batch,
    )
    losses = compute_weighted_loss_terms(
        output=output,
        s_final_batch=s_final_batch,
        domain_head=context.domain_head,
        cfg=cfg,
    )
    losses.total_loss.backward()

    grad_norms = {
        "encoder": _module_grad_norm(context.model.encoder_i) + _module_grad_norm(context.model.encoder_t),
        "hash_head": _module_grad_norm(context.model.hash_head),
        "grl_branch": _module_grad_norm(context.domain_head),
    }

    context.optimizer.step()

    param_delta_norms = {
        "shadow_mainline": _module_param_delta_norm(context.model, before_model),
        "grl_branch": _module_param_delta_norm(context.domain_head, before_grl),
    }

    checks = {
        "batch_has_x_i": "x_i" in batch,
        "batch_has_x_t": "x_t" in batch,
        "batch_has_sample_index": "sample_index" in batch,
        "s_final_batch_shape_ok": s_final_batch.shape == (cfg.smoke.batch_size, cfg.smoke.batch_size),
        "b_is_derived_from_v": bool(
            torch.equal(output.b_i, torch.where(output.v_i >= 0.0, torch.ones_like(output.v_i), -torch.ones_like(output.v_i)).detach())
            and torch.equal(output.b_t, torch.where(output.v_t >= 0.0, torch.ones_like(output.v_t), -torch.ones_like(output.v_t)).detach())
        ),
        "encoder_grad_nonempty": grad_norms["encoder"] > 0.0,
        "hash_head_grad_nonempty": grad_norms["hash_head"] > 0.0,
        "grl_grad_nonempty": grad_norms["grl_branch"] > 0.0,
        "optimizer_step_changed_mainline": param_delta_norms["shadow_mainline"] > 0.0,
        "optimizer_step_changed_grl_branch": param_delta_norms["grl_branch"] > 0.0,
    }

    return TrainerSmokeStepResult(
        dataset=cfg.smoke.dataset,
        feature_set_id=context.dataset.meta.feature_set_id,
        semantic_set_id=cfg.smoke.semantic_set_id,
        batch_rows=int(sample_indices.size),
        sample_index_hash=context.dataset.meta.sample_index_hash,
        sample_indices=sample_indices.astype(int).tolist(),
        losses={
            "loss_sem": float(losses.loss_sem.detach().cpu().item()),
            "loss_q": float(losses.loss_q.detach().cpu().item()),
            "loss_bal": float(losses.loss_bal.detach().cpu().item()),
            "loss_grl": float(losses.loss_grl.detach().cpu().item()),
            "total_loss": float(losses.total_loss.detach().cpu().item()),
        },
        grad_norms=grad_norms,
        param_delta_norms=param_delta_norms,
        checks=checks,
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")


def _current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        raise RuntimeError("optimizer must have at least one parameter group")
    return float(optimizer.param_groups[0]["lr"])


def _write_config_snapshot(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def _checkpoint_model_state_dict(context: TrainingMainlineContext) -> dict[str, Any]:
    return {
        "training_mainline": context.model.state_dict(),
        "domain_head": context.domain_head.state_dict(),
    }


def _save_training_checkpoint(
    *,
    path: Path,
    context: TrainingMainlineContext,
    epoch: int,
    global_step: int,
) -> None:
    payload = {
        "model_state_dict": _checkpoint_model_state_dict(context),
        "optimizer_state_dict": context.optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "feature_set_id": context.dataset.meta.feature_set_id,
        "semantic_set_id": context.config.train.semantic_set_id,
        "sample_index_hash": context.dataset.meta.sample_index_hash,
        "config_key_fields": context.config.checkpoint_key_fields(),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(payload, path)


def _load_training_checkpoint(
    *,
    path: Path,
    context: TrainingMainlineContext,
) -> tuple[int, int]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint at {path} must be a dict payload")

    expected_key_fields = context.config.checkpoint_key_fields()
    actual_key_fields = payload.get("config_key_fields")
    if actual_key_fields != expected_key_fields:
        raise RuntimeError("Checkpoint config key fields do not match current training mainline config")

    if payload.get("feature_set_id") != context.dataset.meta.feature_set_id:
        raise RuntimeError("Checkpoint feature_set_id does not match current feature cache")
    if payload.get("semantic_set_id") != context.config.train.semantic_set_id:
        raise RuntimeError("Checkpoint semantic_set_id does not match current semantic supervision")
    if payload.get("sample_index_hash") != context.dataset.meta.sample_index_hash:
        raise RuntimeError("Checkpoint sample_index_hash does not match current sample index contract")

    model_state_dict = payload.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise RuntimeError("Checkpoint model_state_dict must be a dict")
    if "training_mainline" not in model_state_dict or "domain_head" not in model_state_dict:
        raise RuntimeError("Checkpoint model_state_dict must contain training_mainline and domain_head entries")

    optimizer_state_dict = payload.get("optimizer_state_dict")
    if not isinstance(optimizer_state_dict, dict):
        raise RuntimeError("Checkpoint optimizer_state_dict must be a dict")

    context.model.load_state_dict(model_state_dict["training_mainline"])
    context.domain_head.load_state_dict(model_state_dict["domain_head"])
    context.optimizer.load_state_dict(optimizer_state_dict)

    epoch = payload.get("epoch")
    global_step = payload.get("global_step")
    if not isinstance(epoch, int) or epoch < 0:
        raise RuntimeError("Checkpoint epoch must be non-negative int")
    if not isinstance(global_step, int) or global_step < 0:
        raise RuntimeError("Checkpoint global_step must be non-negative int")
    return epoch, global_step


def _prepare_fresh_run_dir(
    *,
    run_dir: Path,
    config_snapshot: dict[str, Any],
    config_snapshot_path: Path,
    metrics_path: Path,
    checkpoint_path: Path,
) -> None:
    if config_snapshot_path.exists() or metrics_path.exists() or checkpoint_path.exists():
        raise RuntimeError(
            f"Training run directory already contains formal run artifacts: {run_dir}. "
            "Use --resume to continue or choose a different run_name."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_config_snapshot(config_snapshot_path, config_snapshot)


def run_training_mainline(
    cfg: TrainingMainlineConfig,
    *,
    resume: bool = False,
    run_name: str | None = None,
    stop_after_epochs: int | None = None,
) -> TrainingMainlineRunResult:
    context = build_training_mainline_context(cfg)
    run_dir = cfg.resolve_training_run_dir(run_name=run_name)
    config_snapshot_path = run_dir / "config_snapshot.yaml"
    metrics_path = run_dir / "train_metrics.jsonl"
    checkpoint_path = run_dir / "last.pt"

    target_num_epochs = int(cfg.train.num_epochs)
    if stop_after_epochs is not None:
        if stop_after_epochs <= 0:
            raise RuntimeError("stop_after_epochs must be > 0 when provided")
        effective_stop_epoch = min(int(stop_after_epochs), target_num_epochs)
    else:
        effective_stop_epoch = target_num_epochs

    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume requested but checkpoint not found: {checkpoint_path}")
        start_epoch, global_step = _load_training_checkpoint(path=checkpoint_path, context=context)
    else:
        _prepare_fresh_run_dir(
            run_dir=run_dir,
            config_snapshot=cfg.config_snapshot(run_name=run_name),
            config_snapshot_path=config_snapshot_path,
            metrics_path=metrics_path,
            checkpoint_path=checkpoint_path,
        )
        start_epoch = 0
        global_step = 0

    context.model.train()
    context.domain_head.train()

    completed_epoch = start_epoch
    for epoch_index in range(start_epoch, effective_stop_epoch):
        epoch_number = epoch_index + 1
        for batch in context.dataloader:
            sample_index_tensor = batch["sample_index"]
            sample_indices = _extract_sample_indices(sample_index_tensor)
            x_i = batch["x_i"].to(device=context.device, dtype=torch.float32)
            x_t = batch["x_t"].to(device=context.device, dtype=torch.float32)

            context.optimizer.zero_grad(set_to_none=True)
            output = context.model(x_i, x_t)
            s_final_batch = build_s_final_batch(sample_indices=sample_indices, supervision=context.supervision)
            _validate_batch_supervision(
                batch_sample_index=sample_index_tensor,
                sample_indices=sample_indices,
                s_final_batch=s_final_batch,
            )
            losses = compute_weighted_loss_terms(
                output=output,
                s_final_batch=s_final_batch,
                domain_head=context.domain_head,
                cfg=cfg,
            )
            losses.total_loss.backward()
            context.optimizer.step()

            global_step += 1
            metrics = TrainingMainlineStepMetrics(
                epoch=epoch_number,
                global_step=global_step,
                loss_sem=float(losses.loss_sem.detach().cpu().item()),
                loss_q=float(losses.loss_q.detach().cpu().item()),
                loss_bal=float(losses.loss_bal.detach().cpu().item()),
                loss_grl=float(losses.loss_grl.detach().cpu().item()),
                total_loss=float(losses.total_loss.detach().cpu().item()),
                lr=_current_learning_rate(context.optimizer),
            )
            _append_jsonl(metrics_path, metrics.to_dict())

        completed_epoch = epoch_number
        _save_training_checkpoint(
            path=checkpoint_path,
            context=context,
            epoch=completed_epoch,
            global_step=global_step,
        )

    return TrainingMainlineRunResult(
        run_dir=str(run_dir),
        metrics_path=str(metrics_path),
        checkpoint_path=str(checkpoint_path),
        dataset=cfg.train.dataset,
        feature_set_id=context.dataset.meta.feature_set_id,
        semantic_set_id=cfg.train.semantic_set_id,
        sample_index_hash=context.dataset.meta.sample_index_hash,
        resumed_from_checkpoint=resume,
        start_epoch=start_epoch,
        completed_epoch=completed_epoch,
        global_step=global_step,
        target_num_epochs=target_num_epochs,
        stop_after_epochs=stop_after_epochs,
    )
