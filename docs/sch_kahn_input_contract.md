# SCH-KANH Input Contract (Pre-Implementation Freeze)

## 当前阶段冻结边界 (Current Phase Frozen Boundary)
- This document is the interface contract freeze before SCH-KANH implementation.
- Current work scope is interface planning and contract locking only.
- Out of scope:
  - SCH-KANH backbone implementation
  - loss design/implementation
  - training loop implementation
  - evaluation implementation

## Freeze Status Snapshot (2026-04-09)
- Current frozen SCH-KANH implementation status is `forward-only v1` with output stop at `Zout`.
- Frozen v1 deliverables:
  - `Zout_I.npy`
  - `Zout_T.npy`
  - `meta.json`
- Default mainline input baseline remains `X_I / X_T` from feature cache.
- Upstream semantic frozen artifacts remain `S2.npz / S_high.npz / meta.json`; `S2 / S_high` do not auto-enter mainline input.
- `S_graph` is not default-equivalent to `H / L_hat`.
- `Zout -> V -> B` v2 is not in the current frozen scope.

## 主干默认输入契约 (Mainline Default Input Contract)
- SCH-KANH mainline default input baseline is `X_I / X_T` from feature cache.
- `X_I / X_T` keep upstream `sample_index` order as the only default row order basis.
- No semantic cache matrix is a mandatory default input to SCH-KANH in the current phase.

## 上游 semantic 产物职责 (Upstream Semantic Artifact Responsibilities)
- Semantic stage frozen default outputs are:
  - `S2.npz`
  - `S_high.npz`
  - `meta.json`
- `S2 / S_high` are frozen upstream semantic artifacts.
- `S2 / S_high` do not automatically enter SCH-KANH default inputs in the current contract.

## `meta.json` 的校验职责 (`meta.json` Validation Responsibilities)
- `meta.json` is used for interface validation and traceability, not as a trainable matrix input.
- Required responsibilities:
  - lineage tracking (`feature_cache_dir`, manifest linkage)
  - `sample_index_hash` consistency tracking
  - pipeline mode and entrypoint status tracking
  - top-k and sparsification context tracking (`k_candidate`, `k_final`, strategy fields)

## graph-side 输入启用条件 (Graph-Side Input Enablement Conditions)
- If graph-structured input is required in later phases, a new independent graph-side interface must be added.
- Graph-side interface enablement requires explicit contract update, including:
  - input object definition
  - shape/domain/range constraints
  - sample-index alignment constraints
  - validation rules and failure behavior
- Before this explicit interface exists, SCH-KANH must remain on `X_I / X_T` default baseline.

## `S_graph` 与 `H / L_hat` 的非等价结论 (Non-Equivalence Conclusion)
- `S_graph` in semantic cache is a semantic-stage compatibility graph artifact.
- `H / L_hat` in SCH-KANH method is a hypergraph-side operator definition.
- Contract conclusion:
  - `S_graph` is not default-equivalent to `H`
  - `S_graph` is not default-equivalent to `L_hat`
  - direct substitution is prohibited unless a future contract explicitly defines a valid mapping

## 禁止项清单 (Prohibited Items)
- Do not default-reuse `S_high` as graph-side propagation operator input.
- Do not treat `S_graph` as default `H / L_hat`.
- Do not silently change SCH-KANH default input from `X_I / X_T` to semantic matrices.
- Do not start SCH-KANH implementation, loss, training, or evaluation in this contract-freeze round.
- Do not break sample-order/sample-index consistency across stage boundaries.

## 后续进入实现前的门槛 (Preconditions Before Implementation Starts)
- All target datasets have completed semantic `high_only` full-run outputs and validator pass under current contract.
- Interface owners confirm this document as the single baseline for SCH-KANH input boundary.
- If graph-side input is needed, graph-side interface contract must be added and approved first.
- Any implementation PR must reference this contract and declare whether it stays default (`X_I / X_T`) or introduces a new approved graph-side interface.
