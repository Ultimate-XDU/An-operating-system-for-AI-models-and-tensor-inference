# E1 S1 Q4 · DET probe (half-page)

**Date:** 2026-11-03 (UTC+8)  
**Scope:** llama.cpp Q4_K_M side-track only. Not MyOS vs llama exact. Not main-track FP16/BF16.

## Symptom

`runs/s1-llamacpp-q4km-b11960` (**-t 8**, default AMX CPU variant) → **DET-FAIL** (R=5 token ids diverge). Earlier `-t 1 -n 64` on the AMX build also forked.

Stderr on default build: `AMX_INT8 = 1`, `amx repack tensor …`, AMX model buffer ≈950 MiB (Q4_K / Q6_K weights).

## Documented knobs used (no invented env)

| Knob | Source |
|------|--------|
| `OMP_NUM_THREADS=1` | OpenMP (build reports `OPENMP=1`) |
| `GGML_CPU_TILED_MM=0` | `ggml/src/ggml-cpu/tiled/tiled.cpp` |
| `GGML_CPU_DISABLE_FUSION=1` | `ggml/src/ggml-cpu/ggml-cpu.c` |
| Exclude `libggml-cpu-sapphirerapids.so` | only AMX-compiled variant in CMake; `ggml_backend_load_best("cpu")` scores libs next to **real** `llama-cli` path |

No `GGML_DISABLE_AMX` in this tree. Symlinking `llama-cli` keeps searching the original dir (AMX still loads); **copy** the binary into a dir without sapphirerapids.

## Probe table (short → escalate)

| Probe | Config | n_gen | R | DET | Notes |
|-------|--------|------|---|-----|-------|
| P1 | default bin, `-t 1` | 32 | 3 | PASS | still `amx_repack` |
| P1↑ | same | 64 | 3 | PASS | |
| P1↑↑ | same | 128 | 3 | **FAIL** @tok4 | AMX path unstable at S1 length |
| P3 | **noamx dir** + `-t 1` + env trio | 128 | 3 | **PASS** | no AMX feature / no repack |
| Full | P3 + W=3 R=5 | 128 | 5 | **PASS** | → `runs/s1-llamacpp-q4km-b11960-det` |

Raw: `/tmp/det-probes/probe_table.json`.

## DET-stable fingerprint (recorded)

- Path: `/workspace/MyOS/runs/s1-llamacpp-q4km-b11960-det`
- Bin dir: `/workspace/tools/llama.cpp-bin/llama-b11960-noamx`
- `-t 1`, env above, greedy clamps (`temp=0 top-k=1 top-p=1 min-p=0 …`)
- Label for cost numbers: **吞吐/延迟/内存参考**（单线程非 AMX）；**不得**写入副赛道主胜负表，除非协议确认本 fingerprint。副赛道对打在 DET-PASS 可用前不要开；本卡已 DET-PASS，但是否作副赛道基线由协议方拍板。

## DET-FAIL evidence (kept)

- `/workspace/MyOS/runs/s1-llamacpp-q4km-b11960` — multi-thread / AMX default; appendix only; CORRECT/胜负作废。
