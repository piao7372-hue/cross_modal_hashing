# Similarity Matrix Refactor Rules

## Scope
本仓库的相似性矩阵重构与监督图设计必须遵守以下规则。除非任务明确要求做对比实验，否则不要违反这些规则。

## Naming and notation
- 全文只使用 `X_I / X_T` 作为图像和文本特征记号。
- 不要引入 `F_I / F_T` 作为默认实现变量。
- `S_high`、`S_pseudo`、`S_final`、`S_graph` 必须分开命名，不允许混用同一变量承担多个角色。

## Core graph rules
- `S_I = pairwise_cosine_rows(X_I, X_I)`
- `S_T = pairwise_cosine_rows(X_T, X_T)`
- `S2` 中的 `cos(A, B)` 必须表示“矩阵行向量之间的两两 cosine”，不是标量，不是逐元素 cosine。
- 稀疏化采用两阶段：
  - `k_candidate` 用于候选召回
  - `k_final` 用于最终输出
- 不要把第一阶段 top-k 当成最终图。

## Matrix roles
- `S_high` 只作为高维语义候选矩阵 / 监督候选矩阵。
- `S_high` 不兼任图传播矩阵。
- 最终监督矩阵固定为：
  - `S_final = beta * S_high + (1 - beta) * S_pseudo`
- 若需要图传播，则单独构造 `S_graph`，不要复用 `S_high`。

## S_high construction
- 必须按以下顺序构造：
  - `A = rowSoftmax(S_fused / tau)`
  - `A_tilde = A + lambda_loop * I`
  - `S_high = rowNormalize(A_tilde)`
- `softmax -> + self-loop` 之后不能直接结束，必须再做 `rowNormalize`。

## Training alignment
- 连续哈希表示 `V` 只拟合 `S_final`。
- 不要让 `V` 同时拟合 `S_high`、`S_final`、`S_graph` 多个目标。
- `GRL` 放在 `Zout`，不要放在 `V`。

## Engineering requirements
- 保持 sample index 顺序全链路一致，不允许 cache、伪标签、训练映射发生重排。
- 修改后必须补充最少一组数值检查：
  - `S_high` 每行和接近 1
  - `S2` 非退化
  - 两阶段 top-k 的边数符合预期
  - shape 和 dtype 一致
- 若仓库已有测试体系，优先补单元测试或最小数值测试。
- 任何修改都要优先保留现有接口稳定性，除非任务明确要求重构接口。

## Current phase
We are currently only doing semantic similarity.

## Hard boundaries
- Do not enter SCH-KANH main branch.
- Do not modify loss, training, evaluation.
- Do not modify data cleaning or feature extraction frozen outputs.
- Always plan before coding.
- Never use full dense NxN implementation for large datasets.
- Keep sample order and sample_index unchanged.