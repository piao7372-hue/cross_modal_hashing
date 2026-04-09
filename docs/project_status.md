# Project Status

## Freeze Snapshot (2026-04-09)
1. 第二步（semantic similarity 默认主路径 `high_only`）按工程口径已完成并冻结。
2. 三个正式数据集 full-run 均已通过 validator：
   - `nuswide`
   - `mirflickr25k`
   - `mscoco`
3. semantic similarity 当前冻结的默认正式交付为：
   - `S2.npz`
   - `S_high.npz`
   - `meta.json`
4. 第三步（跨模态对齐网络）中的 SCH-KANH 主干 `forward-only v1` 已完成并冻结：
   - smoke 通过
   - `mirflickr25k` full-run 通过
   - `nuswide` full-run 通过
   - `mscoco` full-run 通过
5. SCH-KANH `forward-only v1` 当前冻结交付为：
   - `Zout_I.npy`
   - `Zout_T.npy`
   - `meta.json`
6. 当前尚未正式进入：
   - `Zout -> V -> B` v2
   - graph-side
   - loss
   - training
   - evaluation

## Unified status summary (2026-04 sync)
1. 数据清洗：已完成并冻结。
2. 特征提取：已完成并冻结（feature cache 可用，sample order/sample_index 对齐链路已建立并保持冻结）。
3. semantic similarity：默认主路径已完成一轮契约收口。
4. 当前默认正式输出：`S2.npz`、`S_high.npz`、`meta.json`。
5. 兼容可选输出：`S_graph.npz`（非默认主路径输出）。
6. 可选中间量：`S_I.npz`、`S_T.npz`、`S1.npz`、`S_fused.npz`（非默认正式输出）。
7. 预留但默认未激活：`with_pseudo`、`S_pseudo`、`S_final`。
8. 当前未正式开始：SCH-KANH 主干、loss、training、evaluation、端到端 mAP 验证、最优 k 的最终确定。

## Current phase
Project is currently in semantic similarity contract-closure phase on top of frozen cleaning and feature stages.
Cleaning-only phase statements in this file are historical sub-stage records and not the current whole-project phase definition.
Historical records below do not override the freeze snapshot above.
Out of scope in the current phase: SCH-KANH mainline, loss, training, evaluation, end-to-end mAP validation, final best-k conclusion.

## NUS-WIDE status
NUS-WIDE cleaning is completed and frozen.
Rule: do not modify NUS-WIDE cleaning logic unless a clear bug is found.

## MIRFlickr status
MIRFlickr cleaning first pass is completed and frozen.

Completed and integrated:
1. MIRFlickr cleaner implementation
2. dataset config integration
3. pipeline registration and dispatch

Execution status:
1. `--dataset mirflickr25k --dry-run`: passed
2. `--dataset mirflickr25k` full non-dry-run: passed

Frozen counts:
1. `actual_total_rows = 25000`
2. `actual_clean_rows = 25000`
3. `dropped_records = 0`
4. `drop_reason_breakdown = {}`
5. `empty_text_rows = 2128`
6. `annotation dedup statistics inside label files = 0`

Manual acceptance closure:
1. completed for head / middle / tail samples
2. sampling source: `data/processed/mirflickr25k/clean_manifest.jsonl`
3. sampling basis: current file order
4. fixed positions: line `1 / 12500 / 25000`
5. explicitly not sampled by numeric `image_id` sorting

Dropped-sample manual audit:
1. currently not applicable because full run has `dropped_records = 0`
2. retained as future-triggered / still_open item

Traceability:
1. MIRFlickr completion commit: `b6cf66a`

## Constraints reminder
1. Historical cleaning-stage boundary: Do not start MSCOCO (kept as historical record; not used as current phase judgment).
2. Do not apply paper ra MIRFlickr `<20 labels` filtering as raw cleaning rule.
3. Keep this document as status summary only; do not use it to introduce new cleaning rules or override code/config truth.

# Semantic Similarity 阶段收口总结（默认主路径契约对齐）

## 时间与提交
- 本轮提交：`e808264`
- 提交信息：`refine semantic similarity default cache contract`
- 已推送到远端：`origin/master`

## 本轮目标
本轮不是扩展新算法，也不是进入 SCH-KANH / loss / training / evaluation。  
本轮目标只有一个：

> 对 semantic similarity 阶段的默认主路径做契约收口，使其与当前方法文档主思路一致，并补齐可执行的 validator 门禁。

## 本轮完成内容

### 1. 补齐 semantic cache validator 门禁
已在 `scripts/validate_semantic_cache.py` 中补齐并跑通以下强制检查：

- `S2` 非退化检查
  - `S2.npz` 必须存在
  - shape 为 `N x N`
  - `nnz > 0`
  - 全部值有限
  - `max_abs > 1e-8`
  - `std > 1e-8`
  - `max > min`

- 两阶段 top-k 边数检查
  - `meta.k_candidate` / `meta.k_final` / `meta.stats.candidate_nnz` / `meta.stats.final_nnz` 必须存在
  - `k_candidate >= k_final`
  - `final_nnz == S_high.nnz`
  - `candidate_nnz` 与 `final_nnz` 必须落在理论区间内
  - `candidate_nnz >= final_nnz`
  - `S_high` 每行边数合法
  - `S_high` 对角线严格正
  - `S_high` 行归一检查通过

### 2. 明确 `S2` 为正式默认契约输出
`S2` 已从 `debug_save_intermediates` 语义中剥离，不再依赖 debug 开关。  
当前阶段中，`S2` 是 semantic similarity 默认主路径的正式输出之一。

### 3. 默认正式输出收缩为最小必要集合
当前默认正式输出固定为：

- `S2.npz`
- `S_high.npz`
- `meta.json`

这意味着默认生产路径不再持久化过多中间矩阵，从而降低 full 数据集下的存储、写盘、压缩与归档成本。

### 4. 可选中间量与兼容输出重新定位
以下矩阵不再属于默认正式输出，而是可选中间量：

- `S_I.npz`
- `S_T.npz`
- `S1.npz`
- `S_fused.npz`

`S_graph.npz` 保留为**兼容可选输出**，默认不输出，不再作为当前阶段默认主路径的一部分。

### 5. 文档与配置口径对齐
已完成以下口径对齐：

- `configs/semantic_similarity.yaml`
- `src/semantic_similarity/config.py`
- `src/semantic_similarity/graph_builder.py`
- `src/semantic_similarity/pipeline.py`
- `scripts/validate_semantic_cache.py`
- `docs/semantic_cache_contract.md`
- `AGENTS.md`

其中关键口径包括：

- 当前只做 semantic similarity
- 不进入 SCH-KANH 主干
- 不修改 loss / training / evaluation
- 不动已冻结的数据清洗与特征提取产物
- 不允许 full dense NxN 持久化实现
- 保持 sample order 与 sample_index 不变

## 当前默认主路径定义

### 默认正式输出
- `S2.npz`
- `S_high.npz`
- `meta.json`

### 可选中间量
- `S_I.npz`
- `S_T.npz`
- `S1.npz`
- `S_fused.npz`

### 兼容可选输出
- `S_graph.npz`

## 本轮验证结果

### 1. validator 运行环境确认
使用解释器：

- `C:\Users\ASVS\anaconda3\envs\deeplearning\python.exe`

### 2. 旧缓存契约问题已识别
旧 smoke 缓存 `sem_smoke_mir64_v1` 因缺少 `S2.npz`，在新门禁下失败。  
该失败已确认属于**旧缓存契约问题**，不是环境问题，也不是 validator 逻辑错误。

### 3. 新缓存重产与验证通过
已重新产出并验证：

- `sem_smoke_mir64_v2`：含 `S2` 与更多中间矩阵，validator 通过
- `sem_smoke_mir64_v3`：默认主路径过渡版本，validator 通过
- `sem_smoke_mir64_v4`：最终默认主路径版本，实际产物为  
  - `S2.npz`
  - `S_high.npz`
  - `meta.json`

对 `sem_smoke_mir64_v4` 的 validator 结果：

- `S2` 非退化检查通过
- 两阶段 top-k 边数检查通过
- `S_high` 行归一检查通过
- 可选中间量缺失不报错，符合新契约

### 4. 兼容性回归
历史含 `S_graph` 的缓存 `sem_smoke_mir64_v2` 仍可通过 validator。  
说明兼容路径未被破坏。

## 本轮涉及文件
- `AGENTS.md`
- `configs/semantic_similarity.yaml`
- `docs/semantic_cache_contract.md`
- `scripts/validate_semantic_cache.py`
- `src/semantic_similarity/config.py`
- `src/semantic_similarity/graph_builder.py`
- `src/semantic_similarity/pipeline.py`

## 当前阶段结论
当前 semantic similarity 阶段的**默认主路径契约已经完成收口**。  
可以认为本轮完成了：

1. validator 门禁补齐  
2. 默认输出集合最小化  
3. `S2` / `S_high` / `meta` 主路径确立  
4. 中间量与兼容路径重新分层  
5. 配置、实现、校验、文档口径统一

## 当前明确不做的事
以下内容仍然**不属于当前已完成范围**：

- `S_pseudo` 的正式主路径接入
- `S_final` 的正式训练监督接入
- `S_graph` 的新逻辑扩展
- SCH-KANH 主干接入
- loss / training / evaluation
- 基于验证集 mAP 的最终 k 最优值结论

## 下一阶段建议（只作记录，不立即编码）
下一步应先做**规划审计**，而不是直接改代码。  
建议先审查：

1. 当前仓库中 `S_graph / S_pseudo / S_final / with_pseudo` 的现状
2. 它们是否仍与当前阶段边界冲突
3. 下一轮到底是：
   - 继续只做 semantic similarity 内部职责清理
   - 还是开始规划后续 `S_final` / 主干接入

在下一轮开始前，仍需继续遵守：
- 先计划，后编码
- 一次只做一个阶段
- 不越阶段修改
