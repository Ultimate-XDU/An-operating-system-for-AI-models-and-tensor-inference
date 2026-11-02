# Phase1 对打协议 · MyOS Tensor Runtime vs llama.cpp

**版本**：v0.1（冻结）  
**日期**：2026-11-02（UTC+8）  
**起草**：博弈手  
**对接**：OSdeveloper  
**确认**：月星（按 OSdeveloper 五项暂定拍板冻结）  
**效力**：Phase1 评测执行以本文为准；接口/实现变更若影响公平性须升 v0.2+  
**范围**：Phase1 = WeightLoaderCore + ModelCore + 最小推理循环（prefill + decode）  
**前版**：`PHASE1-ARENA-PROTOCOL-v0.md`（保留作历史）

---

## 冻结决议（v0.1）

| # | 决议 |
|---|------|
| F1 | **MODEL**：主模型 `Qwen2.5-1.5B-Instruct`；SafeTensors 主赛道 **BF16/FP16**；GGUF 副赛道 **Q4_K_M**；内存紧回退 `Qwen2.5-0.5B-Instruct`（回退须全表重跑并注明） |
| F2 | **数值门闩**：Phase1 **只认 exact**（greedy + deterministic=ON）；logits tol 推迟 Phase2 |
| F3 | **S3**：正式表 `n_prompt=4096`；原 8k 记为 **S3+** 附录挑战，不进 Phase1 主胜负 |
| F4 | **赛道**：Phase1 **只打 CPU**（同 `n_threads`）；GPU/CUDA 进 Phase2 附录 |
| F5 | **胜负带宽**：维持 **±5%** |

---

## 0. 假设（写死，否则不对打）

| ID | 假设 |
|----|------|
| A1 | 同一模型权重文件；**SafeTensors 主赛道**与 **GGUF 副赛道分表**，禁止混报宣称胜负 |
| A2 | 主赛道精度 BF16 或 FP16（双方一致）；副赛道量化档 **Q4_K_M**；跨档位禁止直接宣称胜负 |
| A3 | 同一 prompt 字节序列与 tokenizer（或双方约定同一 token 序列文件，禁止「各用各的 tokenize」后比 tokens/s） |
| A4 | 确定性模式开/关必须成对报告；**主表默认 deterministic=ON**（固定 seed、禁非确定 kernel 路径） |
| A5 | 硬件钉死：同一主机、同一 **CPU** 拓扑与 `n_threads`；禁止跨机拼表；Phase1 禁止 GPU 进主表 |
| A6 | Warm-up 不计分：丢弃前 W 次（默认 W=3）；计分 R 次（默认 R=5）取中位数 |
| A7 | MyOS Phase1 目标是 **正确 + 可复现吞吐/延迟/内存**；正确性不过关则吞吐胜负作废 |
| A8 | vLLM / GPU 路径仅附录，**不进 Phase1 主胜负** |
| A9 | 默认模型 `Qwen2.5-1.5B-Instruct`；回退 `0.5B` 时须在 scoreboard 标注 `MODEL_FALLBACK=0.5B` |

---

## 1. 公平对比规则

### 1.1 允许差异（不判犯规）

- 运行时形态：Unikernel / 用户态最小环 vs llama.cpp 进程
- 内存分配器、页表策略、静态链接与否
- 内部算子实现（仍须数值协议见 §2.4）

### 1.2 必须对齐（犯规则本局无效）

| 项 | 对齐方式 |
|----|----------|
| 权重 | 同一文件路径或同一 SHA256 |
| 上下文长度上限 | 双方 `n_ctx` 设为场景规定值 |
| 生成长度 | 场景规定 `n_gen`；不足则记 FAIL |
| Batch | 场景规定 `B`；MyOS 若暂不支持 B>1，该场景标 **N/S**，不记胜 |
| 线程 | 钉同一 `n_threads`（CPU-only） |
| 精度 | 权重精度 + 累计路径写入配置哈希 |
| 输出 | 记录完整配置 JSON + 命令行 + git commit / 镜像 digest |

### 1.3 配置指纹

每次跑必须产出 `run_fingerprint.json`：

```
model_id, model_sha256, fmt, quant_or_dtype, n_ctx, n_prompt, n_gen, batch,
threads, device=cpu, deterministic, seed, commit, cmdline
```

指纹不同 → 禁止并入同一对打表。

---

## 2. 指标定义（主表钉死）

### 2.1 吞吐与延迟

| 符号 | 定义 | 单位 | 主/附 |
|------|------|------|-------|
| **T_decode** | 生成阶段：`n_gen / t_decode`（不含 prefill） | tokens/s | **主** |
| **T_e2e** | `(n_prompt + n_gen) / (t_prefill + t_decode)` | tokens/s | 附 |
| **L_prefill** | 首 token 前耗时（含权重已加载后的 prefill） | ms | **主** |
| **L_TPOT** | decode 阶段平均每 token 时间 = `t_decode / n_gen` | ms | **主** |
| **L_TTFT** | 同 L_prefill（别名，对外文案可用 TTFT） | ms | 同主 |

**计时边界**（必须写进实现）：

1. 权重加载 **单独计** `t_load`，**不进入** L_prefill / T_decode（另报表）。
2. `t_prefill`：从提交 prompt tokens 到产出第一个新 token 之前。
3. `t_decode`：从第一个新 token 到第 `n_gen` 个新 token 完成。

### 2.2 内存

| 符号 | 定义 | 单位 |
|------|------|------|
| **M_peak** | 推理稳态峰值（RSS 或等价 Unikernel 物理页占用）；加载后、生成中取 max | MiB |
| **M_weight** | 权重常驻 | MiB |
| **M_kv** | KV / 工作区估算或实测 | MiB |

主表报 **M_peak**；分解进附录。

### 2.3 确定性

| 开关 | 要求 |
|------|------|
| ON | 同指纹下，R 次 greedy 输出 token 序列 **逐 token 一致**；不一致 → DET-FAIL |
| OFF | 允许非确定；只比吞吐/延迟/内存，**不**做序列一致门闩 |

### 2.4 正确性门闩（胜负前置）· 冻结 F2

在固定 prompt + greedy + deterministic=ON：

1. **与参考实现**（默认 llama.cpp 同权重同赛道）比对：**只认 exact match**（逐 token）。
2. logits / token tol **不进入** Phase1；推迟 Phase2。
3. 门闩失败 → 该场景 **不计吞吐胜负**，记 CORRECT-FAIL。

---

## 3. 对抗场景矩阵（Phase1）

**默认模型**：`Qwen2.5-1.5B-Instruct`（F1）；回退见 A9。

| 场景 ID | 名称 | n_prompt | n_gen | batch B | n_ctx | 主表？ | 目的 |
|---------|------|----------|-------|---------|-------|--------|------|
| S1 | 短上下文·单请求 | 128 | 128 | 1 | ≥512 | 是 | 日常对话延迟/吞吐基线 |
| S2 | 中上下文 | 2048 | 128 | 1 | ≥4096 | 是 | prefill 压力 |
| S3 | 长上下文 | **4096** | 128 | 1 | ≥4096+margin | 是 | 内存墙 + prefill 墙 |
| S3+ | 超长挑战 | 8192 | 128 | 1 | ≥8192+margin | **附录** | 不进 Phase1 主胜负 |
| S4 | 短·微批 | 128 | 128 | 4 | ≥512 | 是（可 N/S） | 批处理效率 |
| S5 | 中·微批 | 2048 | 128 | 4 | ≥4096 | 是（可 N/S） | 批×上下文交叉 |
| S6 | 冷启动加载 | — | — | — | — | 代价表 | 只比 `t_load` + M_weight |

**对抗含义**：

- **延迟对抗**：S1 上比 L_TTFT、L_TPOT
- **吞吐对抗**：S1/S2 上比 T_decode
- **内存对抗**：S3 上比 M_peak；OOM = 该场景 FAIL
- **扩展对抗**：S4/S5；MyOS 不支持则 N/S
- **加载对抗**：S6；不影响推理主胜负

每场景：Warm-up W=3，计分 R=5，主报 **中位数**；附录给 min/max。

---

## 4. 相对 llama.cpp 的胜负判定

### 4.1 对手与角色

| 角色 | 实现 |
|------|------|
| **挑战者** | MyOS Phase1 |
| **主基线** | llama.cpp（同 CPU、同权重赛道） |
| **参考实现** | 同 llama.cpp（exact 正确性） |
| **附录对照** | llama2.c / llama2.rs；宿主机 vLLM；GPU 路径（Phase2） |

### 4.2 单场景判定（CORRECT 通过且非 N/S）· 冻结 F5

| 指标 | 胜（MyOS） | 负 | 平 |
|------|------------|----|----|
| T_decode | ≥ 基线 × **1.05** | ≤ 基线 × **0.95** | 其余 |
| L_TTFT | ≤ 基线 × **0.95** | ≥ 基线 × **1.05** | 其余 |
| L_TPOT | ≤ 基线 × **0.95** | ≥ 基线 × **1.05** | 其余 |
| M_peak | ≤ 基线 × **0.95** | ≥ 基线 × **1.05** | 其余 |

**场景胜负合成**：

- 权重：T_decode **0.35**，L_TPOT **0.25**，L_TTFT **0.20**，M_peak **0.20**
- 胜=+1，平=0，负=−1；加权和 >0 → 场景胜；<0 → 场景负；=0 → 场景平
- 任一 CORRECT-FAIL / OOM / DET-FAIL → **场景负且吞吐分作废**

### 4.3 Phase1 总胜负

| 结局 | 条件 |
|------|------|
| **Phase1 宣称胜** | S1 与 S2 均场景胜，且 S3 非 CORRECT-FAIL/OOM（S3 允许「更慢但省内存」的分项胜，不强制吞吐胜） |
| **Phase1 有条件胜** | S1 胜，S2 平或轻负（加权和 ≥ −0.2），S3 内存胜或平，无正确性失败 |
| **Phase1 负** | S1 负，或任主场景 CORRECT-FAIL |
| **不可比** | 指纹不对齐 / 仅有 N/S / 基线未跑出 |

禁止用 S3+/S4/S5/S6 单独支撑「打败 llama.cpp」对外句。

---

## 5. 代价分析模板（每场景必填）

```text
## Cost Card · {scenario_id} · {date}

### 设定
- fingerprint: …
- model: Qwen2.5-1.5B-Instruct | fallback 0.5B
- track: SafeTensors BF16/FP16 | GGUF Q4_K_M
- device: cpu, n_threads: …
- challenger / baseline commits: …

### 结果（中位数）
| metric | MyOS | llama.cpp | ratio (MyOS/base) | verdict |
|--------|------|-----------|-------------------|---------|
| T_decode |  |  |  | W/T/L |
| L_TTFT |  |  |  | |
| L_TPOT |  |  |  | |
| M_peak |  |  |  | |
| t_load (S6) |  |  |  | |

### 正确性
- greedy exact match: PASS/FAIL
- det consistency: PASS/FAIL

### 代价拆解
- 算力：CPU util 或 cycles（有则填）
- 内存：M_weight / M_kv / 页占用
- 工程：为达该分额外做的特例
- 功能债：N/S 的能力（batch、更长上下文、采样器…）

### 结论一句
- 胜负：…
- 主要代价：…
- 是否值得为 Phase2 继续加码：是/否 + 理由
```

---

## 6. 输出物清单

1. `runs/<fingerprint>/metrics.json`  
2. `runs/<fingerprint>/tokens.txt`（deterministic 跑）  
3. `runs/<fingerprint>/cost-card.md`  
4. 汇总表 `arena/phase1-scoreboard.md`（S1–S6；S3+ 附录）

---

## 7. 已关闭的开放问题

| 原问题 | v0.1 决议 |
|--------|-----------|
| MODEL_X | F1：`Qwen2.5-1.5B-Instruct` + 赛道/回退 |
| exact vs tol | F2：只认 exact |
| S3 长度 | F3：正式 4096；8k→S3+ |
| CPU/GPU | F4：只打 CPU |
| ±5% | F5：维持 |

---

## 8. 下一步实验

| 步 | 内容 | 负责建议 |
|----|------|----------|
| E0 | ✅ 冻结 MODEL + 赛道 + CPU + exact + S3=4k | 月星 / OSdeveloper / 博弈手 |
| E1 | llama.cpp 侧先跑通 S1/S2/S6（及 S3），产出基线 cost card | OSdeveloper |
| E2 | MyOS 最小环 greedy exact 对齐 S1 后开对打 | OSdeveloper |
| E3 | 首轮数字回来后，博弈手视情况提议阈值/场景修订（升 v0.2/v1） | 博弈手 |

---

*路径：`/workspace/MyOS/PHASE1-ARENA-PROTOCOL-v0.1.md`*
