from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .cache_writer import prepare_output_dir, save_array, write_meta
from .chebkan_encoder import ChebKANEncoder
from .config import SchKanhConfig
from .feature_reader import load_feature_cache_inputs
from .hash_head import SharedHashHead


@dataclass(frozen=True)
class MainlineForwardOutput:
    zout_i: np.ndarray
    zout_t: np.ndarray
    v_i: np.ndarray
    v_t: np.ndarray
    b_i: np.ndarray
    b_t: np.ndarray
    rows: int
    dim_in: int
    dim_zout: int
    dim_v: int
    dim_b: int


@dataclass(frozen=True)
class SchKanhStats:
    dataset: str
    feature_set_id: str
    sch_set_id: str
    output_dir: str
    rows: int
    dim_in: int
    dim_zout: int
    dim_v: int
    dim_b: int
    saved_files: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def forward_mainline(x_i: np.ndarray, x_t: np.ndarray, cfg: SchKanhConfig) -> MainlineForwardOutput:
    xi = np.asarray(x_i, dtype=np.float32)
    xt = np.asarray(x_t, dtype=np.float32)
    if xi.ndim != 2 or xt.ndim != 2:
        raise RuntimeError("X_I and X_T must be 2D")
    if xi.shape != xt.shape:
        raise RuntimeError(f"X_I shape {xi.shape} != X_T shape {xt.shape}")

    rows = int(xi.shape[0])
    dim_in = int(xi.shape[1])
    dim_zout = int(cfg.mainline.encoder.d_model)

    # Independent parameters for image/text encoders.
    encoder_i = ChebKANEncoder(
        d_in=dim_in,
        d_model=dim_zout,
        order_k=cfg.mainline.encoder.order_k,
        seed=cfg.runtime.seed,
    )
    encoder_t = ChebKANEncoder(
        d_in=dim_in,
        d_model=dim_zout,
        order_k=cfg.mainline.encoder.order_k,
        seed=cfg.runtime.seed + 1,
    )

    # Locked Zout definition for disabled graph-side.
    zout_i = encoder_i.forward(xi)
    zout_t = encoder_t.forward(xt)

    # Locked v2 hash-head definition with shared parameters.
    hash_head = SharedHashHead(
        d_in=dim_zout,
        d_hash=cfg.mainline.hash_head.d_hash,
        seed=cfg.runtime.seed + 2,
    )
    hash_out = hash_head.forward(zout_i, zout_t)

    return MainlineForwardOutput(
        zout_i=zout_i,
        zout_t=zout_t,
        v_i=hash_out.v_i,
        v_t=hash_out.v_t,
        b_i=hash_out.b_i,
        b_t=hash_out.b_t,
        rows=rows,
        dim_in=dim_in,
        dim_zout=dim_zout,
        dim_v=hash_out.d_hash,
        dim_b=hash_out.d_hash,
    )


def run_sch_kahn_mainline(
    *,
    dataset: str,
    feature_set_id: str,
    sch_set_id: str,
    config: SchKanhConfig,
    overwrite: bool,
) -> SchKanhStats:
    if dataset not in {"nuswide", "mirflickr25k", "mscoco"}:
        raise ValueError(f"Unsupported dataset: {dataset}")

    feature_cache_dir = config.resolve_feature_cache_dir(dataset=dataset, feature_set_id=feature_set_id)
    features = load_feature_cache_inputs(feature_cache_dir)

    out = forward_mainline(features.x_i, features.x_t, config)

    output_dir = config.resolve_sch_kahn_cache_dir(dataset=dataset, sch_set_id=sch_set_id)
    prepare_output_dir(output_dir, overwrite=overwrite)

    save_array(output_dir / "Zout_I.npy", out.zout_i)
    save_array(output_dir / "Zout_T.npy", out.zout_t)
    save_array(output_dir / "V_I.npy", out.v_i)
    save_array(output_dir / "V_T.npy", out.v_t)
    save_array(output_dir / "B_I.npy", out.b_i)
    save_array(output_dir / "B_T.npy", out.b_t)

    meta = {
        "contract_version": "sch_kahn_mainline_cache_v2",
        "dataset": dataset,
        "feature_set_id": features.feature_set_id,
        "sch_set_id": sch_set_id,
        "input_source": "feature_cache_x_i_x_t",
        "sample_index_hash": features.sample_index_hash,
        "graph_side": {
            "mode": config.mainline.graph_side.mode,
        },
        "stop_at": config.mainline.stop_at,
        "rows": out.rows,
        "dim": {
            "in": out.dim_in,
            "zout": out.dim_zout,
            "v": out.dim_v,
            "b": out.dim_b,
        },
        "hash_head": {
            "enabled": True,
            "shared_params": config.mainline.hash_head.shared_params,
            "binarize_rule": config.mainline.hash_head.binarize_rule,
            "d_hash": config.mainline.hash_head.d_hash,
        },
        "dtype": config.runtime.dtype,
        "device": config.runtime.device,
        "seed": config.runtime.seed,
        "lineage": {
            "feature_cache_dir": features.feature_cache_dir.as_posix(),
            "feature_set_id": features.feature_set_id,
            "sample_index_hash": features.sample_index_hash,
        },
        # Explicitly document forward-only boundary: no semantic cache matrices are consumed.
        "semantic_inputs_used": [],
        "output": {
            "dir": output_dir.as_posix(),
            "saved_files": [
                "Zout_I.npy",
                "Zout_T.npy",
                "V_I.npy",
                "V_T.npy",
                "B_I.npy",
                "B_T.npy",
                "meta.json",
            ],
        },
    }
    write_meta(output_dir / "meta.json", meta)

    return SchKanhStats(
        dataset=dataset,
        feature_set_id=features.feature_set_id,
        sch_set_id=sch_set_id,
        output_dir=output_dir.as_posix(),
        rows=out.rows,
        dim_in=out.dim_in,
        dim_zout=out.dim_zout,
        dim_v=out.dim_v,
        dim_b=out.dim_b,
        saved_files=[
            "Zout_I.npy",
            "Zout_T.npy",
            "V_I.npy",
            "V_T.npy",
            "B_I.npy",
            "B_T.npy",
            "meta.json",
        ],
    )
