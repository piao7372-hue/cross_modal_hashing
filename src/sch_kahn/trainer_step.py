from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .loss_input_builder import build_s_final_batch
from .semantic_supervision_reader import load_semantic_supervision
from .torch_losses import GrlDomainHead, WeightedLossTerms, loss_bal, loss_grl, loss_q, loss_sem
from .torch_mainline import TorchMainlineOutput, TrainerSmokeShadowMainline
from .trainer_config import TrainerSmokeConfig
from .trainer_dataset import FeatureCacheTrainerDataset


@dataclass
class TrainerSmokeContext:
    config: TrainerSmokeConfig
    dataset: FeatureCacheTrainerDataset
    dataloader: DataLoader[dict[str, torch.Tensor]]
    supervision: Any
    model: TrainerSmokeShadowMainline
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


def _resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested torch device `{device_name}` but CUDA is not available")
    return torch.device(device_name)


def _build_optimizer(cfg: TrainerSmokeConfig, context_params: list[torch.nn.Parameter]) -> torch.optim.Optimizer:
    if cfg.optimizer.name == "sgd":
        return torch.optim.SGD(context_params, lr=cfg.optimizer.lr)
    raise RuntimeError(f"Unsupported trainer smoke optimizer: {cfg.optimizer.name}")


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


def compute_weighted_loss_terms(
    *,
    output: TorchMainlineOutput,
    s_final_batch: Any,
    domain_head: GrlDomainHead,
    cfg: TrainerSmokeConfig,
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


def run_single_trainer_smoke_step(cfg: TrainerSmokeConfig) -> TrainerSmokeStepResult:
    context = build_trainer_smoke_context(cfg)
    batch = _fetch_single_batch(context)
    sample_index_tensor = batch["sample_index"]
    sample_indices = sample_index_tensor.cpu().numpy().astype(np.int64, copy=False)
    x_i = batch["x_i"].to(device=context.device, dtype=torch.float32)
    x_t = batch["x_t"].to(device=context.device, dtype=torch.float32)

    before_model = _module_param_snapshot(context.model)
    before_grl = _module_param_snapshot(context.domain_head)

    context.optimizer.zero_grad(set_to_none=True)
    output = context.model(x_i, x_t)
    s_final_batch = build_s_final_batch(sample_indices=sample_indices, supervision=context.supervision)
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
