# E1 · llama.cpp S1 main-track (self-converted BF16 GGUF)

**Date:** 2026-11-03 (UTC+8)  
**Fingerprint id / run dir:** `/workspace/MyOS/runs/s1-llamacpp-bf16-b11960`  
**Track:** `main-track-gguf-bf16` — self-converted BF16 GGUF  
**Role:** **PREFERRED** llama.cpp baseline for later E2 exact vs MyOS ST-BF16 (same dtype family).  
**Do not overwrite** FP16 run dir `runs/s1-llamacpp-fp16-b11960` (kept as parallel FP16 main-track).

## Artifacts

| File | Role |
|------|------|
| `run_fingerprint.json` | §1.3 fingerprint |
| `tokens.txt` | common generated text (DET-PASS) |
| `token_ids.txt` | common next-token id list (128 ids) |
| `metrics.json` | W=3 discarded + R=5 scored; medians |
| `cost-card.md` | human summary |
| `det_evidence.json` | all scored texts + token id lists |

Raw per-run stderr/stdout/ids: `/tmp/e1-s1-bf16-raw/`

## Model / binary

| Item | Value |
|------|--------|
| GGUF | `/workspace/MyOS/weights/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-bf16.gguf` |
| SHA256 | `1a32a39cefefe73f82b4600ccecfe4f3f9411337543a7481f1aa3330d71c973b` |
| Size | ≈ 2.88 GB (BF16) |
| Source | `convert_hf_to_gguf.py --outtype bf16` from `/workspace/MyOS/weights/Qwen2.5-1.5B-Instruct` via `/workspace/tools/venv-gguf` |
| Binary | `/workspace/tools/llama.cpp-bin/llama-b11960/llama-cli` (stock; **not** noamx) |
| Build | b11960 / commit `8642541b255bfb5dc39cd351a178be09013a55d6` |
| `LD_LIBRARY_PATH` | `/workspace/tools/llama.cpp-bin/llama-b11960` |
| Threads | **8** (`-t 8`) |

## Prompt

- File: `/workspace/MyOS/arena/prompts/s1_prompt.txt` — **user message body only** (UTF-8)
- llama-cli default Qwen jinja + system: `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`
- **Measured `prompt_n=128`** on all W+R runs (`slot print_timing`)

## Threads / cmdline

- Pinned **threads=8** (same as FP16 main-track)
- Greedy clamps: `--temp 0 --top-k 1 --top-p 1.0 --min-p 0.0 --typical 1.0 --repeat-penalty 1.0 --presence-penalty 0 --frequency-penalty 0 --dry-multiplier 0 -s 0`
- See `run_fingerprint.json` → `cmdline`
- Runner: `/tmp/e1_s1_bf16_runner.py` (adapted from FP16; env-overridable BIN/LIB/THREADS)

## Medians (R=5)

| metric | median |
|--------|--------|
| T_decode (tok/s) | 17.2348 |
| L_TTFT / L_prefill (ms) | 608.110 |
| L_TPOT (ms) | 58.022 |
| M_peak (MiB RSS) | 3050.953 |
| T_e2e (tok/s) | 31.8608 |

## DET

- **DET-PASS** at fingerprint `-t 8` across R=5: `ids_identical=True`, `text_identical=True`, lens=`[128,128,128,128,128]`, `first_diff=null`.
- Token ids from llama-cli `-v` stderr `next token: <id>` lines.
- **No** `s1-llamacpp-bf16-b11960-det` / noamx `-t 1` appendix needed (default `-t 8` already closed).

## Protocol status

- E1 main-track BF16 baseline for this fingerprint is **closed** (DET-PASS).
- Preferred pairing for E2 exact vs MyOS ST-BF16.
- E2 vs this fingerprint: **CORRECT: PASS** on 2026-11-03 (see `arena/e2-s1-myos-exact.md`, MyOS `42acbb39`).
- 正式账本：`arena/e2-s1-correct.md`；总览：`arena/phase1-scoreboard.md`；可跟踪 cost card：`arena/e1-s1-bf16-cost-card.md`（`runs/` 内 `cost-card.md` 仍不入库）。
- Q4 side-track remains appendix-only; FP16 main-track remains available with dtype caveat.
