# Semantic Cache Contract (Stage: Semantic Similarity / Semantic Graph / Supervision Matrix)

## Scope
This document defines only the semantic matrix layer between feature cache and downstream backbone/loss.
Out of scope:
1. SCH-KANH backbone implementation
2. loss/training/evaluation implementation

## Input Contract
1. Engineering inputs are `X_I` and `X_T` from feature cache.
2. If a document uses `F_I/F_T`, treat them as aliases of `X_I/X_T`.
3. Sample order must strictly follow upstream `sample_index` order.

## Fixed Matrix Roles
1. `S_I`, `S_T`: intra-modal base similarities
2. `S1`: intra-modal fusion
3. `S2`: cross-modal direction-aware term
4. `S_fused`: denoised high-dimensional fusion
5. `S_high`: high-dimensional semantic candidate matrix
6. `S_pseudo`: low-dimensional pseudo-structure matrix
7. `S_final`: final supervision matrix (only in `with_pseudo`)
8. `S_graph`: propagation graph matrix

## Fixed Formulas
1. `S_I(i,j) = cosine(X_I[i,:], X_I[j,:])`
2. `S_T(i,j) = cosine(X_T[i,:], X_T[j,:])`
3. `S1 = alpha1 * S_I + (1 - alpha1) * S_T`
4. `S2 = alpha3 * ( alpha1 * cos(S_I, S_I) + (1 - alpha1) * cos(S_T, S_T) ) + (1 - alpha3) * ( alpha2 * cos(S_T, S_I) + (1 - alpha2) * cos(S_I, S_T) )`
5. `S_fused = tanh( S1 + S2 ⊙ sigmoid(S2) )`
6. `S_high = rowNormalize( rowSoftmax(S_fused / tau) + lambda * I )`
7. `A = exp(S_fused / tau)`, `A_sym = (A + A^T) / 2`, `A_tilde = A_sym + lambda * I`, `S_graph = D^(-1/2) * A_tilde * D^(-1/2)`
8. `S_final = beta * S_high + (1 - beta) * S_pseudo`

## S2 cos(A,B) Semantics (Must Be Identical in Code)
`cos(A, B)` is row-wise relation-vector cosine in global sample-index space:
`[cos(A,B)]_ij = cosine(A[i,:], B[j,:])`.
Sparse implementation is only an acceleration of this dense definition; it must not change the semantic definition into local-support cosine.

## Two-Stage Top-k (Mandatory)
1. Stage 1: build wide candidates from `S_I` and `S_T` with `k_candidate`.
2. Stage 2: compute `S1/S2/S_fused` on candidate union, then keep `k_final`.
3. Constraints:
4. `k_candidate >= k_final`
5. each row must keep self-loop
6. if candidate count < `k_final`, keep actual edges without random fill
7. `k_candidate/k_final` must be dataset-specific in config

## Sparse Softmax Domain
1. Large-scale default domain is `candidate_subgraph`.
2. Row-softmax domain is `candidate edges + self-loop`.
3. `meta.json` must record `softmax_domain = candidate_subgraph`.
4. `full_row` is allowed only for small dense debug mode.

## Symmetrization Rule for S_graph
1. Rule is fixed: undirected union + arithmetic mean = `(A + A^T) / 2`.
2. Do not use `max/sum/first-seen` alternatives.
3. After symmetrization, do not apply another top-k pruning.

## Pipeline Modes
### `high_only`
Outputs:
1. `S_high.npz`
2. `S_graph.npz`
3. `meta.json`
4. optional intermediates (`S_I/S_T/S1/S2/S_fused`) only when `debug_save_intermediates=true`

Meta requirements:
1. `pipeline_mode = high_only`
2. `entrypoints.supervision_target = unavailable`
3. `entrypoints.propagation_graph = S_graph`

### `with_pseudo`
Outputs (additional):
1. `S_pseudo.npz`
2. `S_final.npz`

Meta requirements:
1. `pipeline_mode = with_pseudo`
2. `entrypoints.supervision_target = S_final`
3. `entrypoints.propagation_graph = S_graph`
4. include pseudo-source fields (for example `pseudo_source_mode`, `z_source`, `clustering_method`, `n_clusters`, `pseudo_seed`)

Fail-fast rule:
If pseudo source is unconfigured/incomplete, `with_pseudo` must fail fast and must not write `S_pseudo/S_final`.

## Output Directory
`data/processed/<dataset>/semantic_cache/<semantic_set_id>/`

Primary matrix format:
1. CSR `.npz` for large-scale outputs
2. no dense full matrix default on large datasets

## meta.json Is the Single Source of Truth
Do not maintain a second role/target mapping file.
`meta.json` must include at least:
1. `feature_set_id`
2. `sample_index_hash`
3. lineage fields from upstream feature cache (manifest path/sha or equivalent)
4. `formula_version`
5. `pipeline_mode`
6. `softmax_domain`
7. `symmetrization_rule`
8. `sparsification_strategy`
9. `k_candidate`
10. `k_final`
11. `dtype`
12. `device`
13. `seed`
14. `role_map`
15. `entrypoints`

## Lambda Naming
Config keeps key `lambda` to match formulas.
Code uses `lambda_self_loop` as the internal field name.
