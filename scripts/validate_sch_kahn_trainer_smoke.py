from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sch_kahn.loss_input_builder import build_s_final_batch  # noqa: E402
from sch_kahn.semantic_supervision_reader import load_semantic_supervision  # noqa: E402
from sch_kahn.torch_losses import loss_grl, loss_sem  # noqa: E402
from sch_kahn.trainer_config import load_trainer_smoke_config  # noqa: E402
from sch_kahn.trainer_step import (  # noqa: E402
    _fetch_single_batch,
    build_trainer_smoke_context,
    run_single_trainer_smoke_step,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SCH-KANH trainer smoke shadow path.")
    parser.add_argument("--config", default="configs/sch_kahn_trainer_smoke.yaml")
    return parser.parse_args()


def _assert_signature_exact(fn: Any, *, expected_names: list[str]) -> bool:
    actual = [param.name for param in inspect.signature(fn).parameters.values()]
    return actual == expected_names


def _snapshot_sch_cache_tree(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def _build_bad_entrypoint_fixture(*, semantic_dir: Path) -> Path:
    tmp_dir = PROJECT_ROOT / "outputs" / "tmp_bad_semantic_entrypoint_for_trainer_validation"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    with (semantic_dir / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["entrypoints"]["supervision_target"] = "S_high"

    shutil.copy2(semantic_dir / "S_final.npz", tmp_dir / "S_final.npz")
    with (tmp_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return tmp_dir


def _run_grl_only_check(cfg_path: Path) -> tuple[bool, float, float, float]:
    cfg = load_trainer_smoke_config(PROJECT_ROOT, cfg_path)
    context = build_trainer_smoke_context(cfg)
    batch = _fetch_single_batch(context)
    x_i = batch["x_i"].to(device=context.device, dtype=torch.float32)
    x_t = batch["x_t"].to(device=context.device, dtype=torch.float32)

    context.optimizer.zero_grad(set_to_none=True)
    output = context.model(x_i, x_t)
    grl_only = loss_grl(output.zout_i, output.zout_t, context.domain_head)
    grl_only.backward()

    encoder_grad = sum(
        float(param.grad.detach().abs().sum().item())
        for module in [context.model.encoder_i, context.model.encoder_t]
        for param in module.parameters()
        if param.grad is not None
    )
    hash_head_grad = sum(
        float(param.grad.detach().abs().sum().item())
        for param in context.model.hash_head.parameters()
        if param.grad is not None
    )
    domain_grad = sum(
        float(param.grad.detach().abs().sum().item())
        for param in context.domain_head.parameters()
        if param.grad is not None
    )
    return encoder_grad > 0.0, hash_head_grad, domain_grad, float(grl_only.detach().cpu().item())


def main() -> int:
    args = parse_args()
    cfg_path = (PROJECT_ROOT / args.config).resolve()
    cfg = load_trainer_smoke_config(PROJECT_ROOT, cfg_path)

    sch_cache_root = cfg.processed_root / cfg.smoke.dataset / "sch_kahn_cache"
    tree_before = _snapshot_sch_cache_tree(sch_cache_root)
    step_result = run_single_trainer_smoke_step(cfg)
    tree_after = _snapshot_sch_cache_tree(sch_cache_root)

    context = build_trainer_smoke_context(cfg)
    batch = _fetch_single_batch(context)
    sample_indices = batch["sample_index"].cpu().numpy()
    x_i = batch["x_i"].to(device=context.device, dtype=torch.float32)
    x_t = batch["x_t"].to(device=context.device, dtype=torch.float32)
    output = context.model(x_i, x_t)
    s_final_batch = build_s_final_batch(sample_indices=sample_indices, supervision=context.supervision)
    sem_value = loss_sem(output.v_i, output.v_t, s_final_batch)
    zero_target = s_final_batch.copy()
    zero_target.data[:] = 0.0
    zero_target.eliminate_zeros()
    sem_value_zero = loss_sem(output.v_i, output.v_t, zero_target)

    global_dense_blocked = False
    try:
        context.supervision.s_final.toarray()
    except RuntimeError:
        global_dense_blocked = True

    bad_entrypoint_dir = _build_bad_entrypoint_fixture(semantic_dir=context.supervision.semantic_cache_dir)
    s_high_fallback_blocked = False
    try:
        load_semantic_supervision(
            bad_entrypoint_dir,
            expected_sample_index_hash=context.supervision.sample_index_hash,
            expected_rows=context.supervision.rows,
            expected_feature_set_id=context.supervision.feature_set_id,
        )
    except RuntimeError:
        s_high_fallback_blocked = True
    finally:
        if bad_entrypoint_dir.exists():
            shutil.rmtree(bad_entrypoint_dir)

    grl_encoder_ok, grl_hash_head_grad, grl_branch_grad, grl_only_value = _run_grl_only_check(cfg_path)

    checks = {
        "trainer_step_checks_ok": all(step_result.checks.values()),
        "loss_sem_signature_ok": _assert_signature_exact(
            loss_sem,
            expected_names=["v_i", "v_t", "s_final_batch"],
        ),
        "loss_sem_depends_on_s_final_batch": abs(float(sem_value.detach().cpu().item()) - float(sem_value_zero.detach().cpu().item())) > 1.0e-8,
        "loss_grl_signature_ok": _assert_signature_exact(
            loss_grl,
            expected_names=["zout_i", "zout_t", "domain_head"],
        ),
        "loss_grl_zout_only": grl_encoder_ok and grl_hash_head_grad == 0.0 and grl_branch_grad > 0.0,
        "s_high_fallback_blocked": s_high_fallback_blocked,
        "global_dense_blocked": global_dense_blocked,
        "mainline_cache_tree_unchanged": tree_before == tree_after,
        "encoder_grad_nonempty": step_result.grad_norms["encoder"] > 0.0,
        "hash_head_grad_nonempty": step_result.grad_norms["hash_head"] > 0.0,
        "grl_branch_grad_nonempty": step_result.grad_norms["grl_branch"] > 0.0,
    }
    for key, passed in checks.items():
        if not passed:
            raise RuntimeError(f"Trainer smoke validation failed at check: {key}")

    result = {
        "step_result": step_result.to_dict(),
        "grl_only_loss": grl_only_value,
        "checks": checks,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
