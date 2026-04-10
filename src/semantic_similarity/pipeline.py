from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scipy import sparse

from .cache_writer import avg_degree_of, density_of, prepare_output_dir, save_sparse_npz, write_meta
from .config import SemanticSimilarityConfig
from .feature_reader import load_feature_cache_inputs
from .graph_builder import build_graph_matrices
from .pseudo_builder import PseudoBuildResult, PseudoLabelBuilder
from .supervision_merger import merge_supervision


@dataclass(frozen=True)
class SemanticStats:
    dataset: str
    feature_set_id: str
    semantic_set_id: str
    pipeline_mode: str
    output_dir: str
    rows: int
    dim: int
    saved_matrices: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def _build_role_map_and_entrypoints(
    mode: str,
    *,
    save_graph_compat_output: bool,
) -> tuple[dict[str, str], dict[str, str]]:
    if mode == "high_only":
        role_map = {
            "S_I": "intra_modal_image_base",
            "S_T": "intra_modal_text_base",
            "S1": "intra_modal_fusion",
            "S2": "cross_modal_direction_aware",
            "S_fused": "high_dim_fused",
            "S_high": "semantic_candidate",
        }
        if save_graph_compat_output:
            role_map["S_graph"] = "propagation_graph"
        entrypoints = {
            "supervision_target": "unavailable",
            "propagation_graph": "S_graph" if save_graph_compat_output else "unavailable",
        }
        return role_map, entrypoints
    if mode == "with_pseudo":
        role_map = {
            "S_I": "intra_modal_image_base",
            "S_T": "intra_modal_text_base",
            "S1": "intra_modal_fusion",
            "S2": "cross_modal_direction_aware",
            "S_fused": "high_dim_fused",
            "S_high": "semantic_candidate",
            "S_pseudo": "pseudo_structure",
            "S_final": "supervision_target",
        }
        if save_graph_compat_output:
            role_map["S_graph"] = "propagation_graph"
        entrypoints = {
            "supervision_target": "S_final",
            "propagation_graph": "S_graph" if save_graph_compat_output else "unavailable",
        }
        return role_map, entrypoints
    raise RuntimeError(f"Unsupported mode: {mode}")



def run_semantic_similarity(
    *,
    dataset: str,
    feature_set_id: str,
    semantic_set_id: str,
    config: SemanticSimilarityConfig,
    overwrite: bool,
) -> SemanticStats:
    if dataset not in {"nuswide", "mirflickr25k", "mscoco"}:
        raise ValueError(f"Unsupported dataset: {dataset}")

    feature_cache_dir = config.resolve_feature_cache_dir(dataset=dataset, feature_set_id=feature_set_id)
    features = load_feature_cache_inputs(feature_cache_dir)

    topk_cfg = config.semantic.topk_by_dataset[dataset]

    pseudo_builder = PseudoLabelBuilder(config.pseudo)
    pseudo_result: PseudoBuildResult | None = None
    if config.semantic.pipeline_mode == "with_pseudo":
        # Fail-fast for with_pseudo mode before writing any semantic outputs.
        pseudo_result = pseudo_builder.build(features=features, runtime_cfg=config.runtime)

    graph_result = build_graph_matrices(
        x_i=features.x_i,
        x_t=features.x_t,
        topk_cfg=topk_cfg,
        semantic_cfg=config.semantic,
        runtime_cfg=config.runtime,
    )

    matrices: dict[str, sparse.csr_matrix] = dict(graph_result.matrices)
    pseudo_meta: dict[str, Any] | None = None
    if config.semantic.pipeline_mode == "with_pseudo":
        if pseudo_result is None:
            raise RuntimeError("with_pseudo mode requires pseudo result")
        matrices["S_pseudo"] = pseudo_result.s_pseudo
        matrices["S_final"] = merge_supervision(
            s_high=matrices["S_high"],
            s_pseudo=matrices["S_pseudo"],
            beta=config.semantic.params.beta,
        )
        pseudo_meta = pseudo_result.metadata

    output_dir = config.resolve_semantic_cache_dir(dataset=dataset, semantic_set_id=semantic_set_id)
    prepare_output_dir(output_dir=output_dir, overwrite=overwrite)

    saved: list[str] = []

    # Current phase formal outputs (default path): minimal contract for semantic stage.
    formal_output_keys = ["S2", "S_high"]
    for key in formal_output_keys:
        if key not in matrices:
            raise RuntimeError(f"Missing required formal semantic matrix: {key}")
        path = output_dir / f"{key}.npz"
        save_sparse_npz(path, matrices[key])
        saved.append(path.name)

    # Optional intermediate outputs for debugging/auditing only.
    if config.semantic.debug_save_intermediates:
        for key in ["S_I", "S_T", "S1", "S_fused"]:
            if key not in matrices:
                raise RuntimeError(
                    f"debug_save_intermediates=true requires intermediate matrix: {key}"
                )
            path = output_dir / f"{key}.npz"
            save_sparse_npz(path, matrices[key])
            saved.append(path.name)

    # Compatibility-only output for propagation graph in current phase.
    if config.semantic.save_graph_compat_output:
        if "S_graph" not in matrices:
            raise RuntimeError("save_graph_compat_output=true requires S_graph matrix")
        path = output_dir / "S_graph.npz"
        save_sparse_npz(path, matrices["S_graph"])
        saved.append(path.name)

    if config.semantic.pipeline_mode == "with_pseudo":
        # with_pseudo writes both supervision matrices only when pseudo source is fully ready.
        for key in ["S_pseudo", "S_final"]:
            if key not in matrices:
                raise RuntimeError(f"Missing required matrix in with_pseudo mode: {key}")
            path = output_dir / f"{key}.npz"
            save_sparse_npz(path, matrices[key])
            saved.append(path.name)

    role_map, entrypoints = _build_role_map_and_entrypoints(
        config.semantic.pipeline_mode,
        save_graph_compat_output=config.semantic.save_graph_compat_output,
    )

    stats_obj: dict[str, Any] = {
        "rows": features.rows,
        "dim": features.dim,
        "candidate_nnz": graph_result.stats.get("candidate_nnz"),
        "final_nnz": graph_result.stats.get("final_nnz"),
        "density": {
            "S_high": density_of(matrices["S_high"]),
        },
        "avg_degree": {
            "S_high": avg_degree_of(matrices["S_high"]),
        },
    }
    if config.semantic.save_graph_compat_output:
        stats_obj["s_graph_nnz"] = graph_result.stats.get("s_graph_nnz")
        stats_obj["density"]["S_graph"] = density_of(matrices["S_graph"])
        stats_obj["avg_degree"]["S_graph"] = avg_degree_of(matrices["S_graph"])
    if config.semantic.pipeline_mode == "with_pseudo":
        stats_obj["density"]["S_pseudo"] = density_of(matrices["S_pseudo"])
        stats_obj["density"]["S_final"] = density_of(matrices["S_final"])
        stats_obj["avg_degree"]["S_pseudo"] = avg_degree_of(matrices["S_pseudo"])
        stats_obj["avg_degree"]["S_final"] = avg_degree_of(matrices["S_final"])

    meta: dict[str, Any] = {
        "contract_version": "semantic_cache_v1",
        "dataset": dataset,
        "feature_set_id": features.feature_set_id,
        "semantic_set_id": semantic_set_id,
        "formula_version": config.semantic.formula_version,
        "pipeline_mode": config.semantic.pipeline_mode,
        "lineage": {
            "feature_cache_dir": features.feature_cache_dir.as_posix(),
            "feature_set_id": features.feature_set_id,
            "sample_index_hash": features.sample_index_hash,
            "manifest_path": features.lineage.get("manifest_path"),
            "manifest_sha256": features.lineage.get("manifest_sha256"),
            "manifest_rows": features.lineage.get("manifest_rows"),
            "sample_index_basis": features.lineage.get("sample_index_basis"),
        },
        "sample_index_hash": features.sample_index_hash,
        "softmax_domain": config.semantic.softmax_domain,
        "symmetrization_rule": config.semantic.symmetrization_rule,
        "sparsification_strategy": config.semantic.sparsification_strategy,
        "k_candidate": topk_cfg.k_candidate,
        "k_final": topk_cfg.k_final,
        "dtype": config.runtime.dtype,
        "device": config.runtime.device,
        "seed": config.runtime.seed,
        "params": {
            "alpha1": config.semantic.params.alpha1,
            "alpha2": config.semantic.params.alpha2,
            "alpha3": config.semantic.params.alpha3,
            "beta": config.semantic.params.beta,
            "tau": config.semantic.params.tau,
            "lambda": config.semantic.params.lambda_self_loop,
            "lambda_code_field": "lambda_self_loop",
        },
        "role_map": role_map,
        "entrypoints": entrypoints,
        "output": {
            "dir": output_dir.as_posix(),
            "matrix_format": "csr_npz",
            "saved_files": saved + ["meta.json"],
            "debug_save_intermediates": config.semantic.debug_save_intermediates,
            "save_graph_compat_output": config.semantic.save_graph_compat_output,
        },
        "stats": stats_obj,
    }
    if pseudo_meta is not None:
        meta.update(pseudo_meta)

    write_meta(output_dir / "meta.json", meta)
    saved.append("meta.json")

    return SemanticStats(
        dataset=dataset,
        feature_set_id=features.feature_set_id,
        semantic_set_id=semantic_set_id,
        pipeline_mode=config.semantic.pipeline_mode,
        output_dir=output_dir.as_posix(),
        rows=features.rows,
        dim=features.dim,
        saved_matrices=saved,
    )

