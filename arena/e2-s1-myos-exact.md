正式账本见 `e2-s1-correct.md`。

# E2 · MyOS vs llama.cpp BF16 S1 exact

**Date:** 2026-11-03 (UTC+8)  
**MyOS run dir:** `/workspace/MyOS/runs/e2-myos-s1` (gitignored)  
**Ref run dir:** `/workspace/MyOS/runs/s1-llamacpp-bf16-b11960`  
**Track:** main-track ST-BF16 (MyOS) vs GGUF-BF16 (llama.cpp)  
**Gate:** `--ref-ids` → **`CORRECT: PASS`** (128/128 token ids identical)

## Verdict

| Gate | Result |
|------|--------|
| CORRECT | **PASS** |
| first_diff | none |
| n_gen compared | 128 |
| MyOS commit | `42acbb3920c597ab31e61f72e4cf90502b915584` (PR #27) |

E2 S1 exact on the preferred BF16 pairing is **closed**. Throughput / cost card comparison is **not** claimed here: MyOS wall ≈31 min (naive `bf16_w_fp32_acc`, `threads=1`); llama.cpp BF16 median wall ≈8.8 s (`-t 8`). Per protocol A7, CORRECT PASS unlocks using the E1 BF16 medians as the llama reference; MyOS speed is a separate optimization track.

## Artifacts (local, gitignored under `runs/`)

| File | Role |
|------|------|
| `runs/e2-myos-s1/run_fingerprint.json` | MyOS §1.3 fingerprint (`fmt=safetensors`, `quant_or_dtype=bf16`, `acc_path=bf16_w_fp32_acc`, `n_prompt=128`, `n_gen=128`, `n_ctx=256`, `threads=1`) |
| `runs/e2-myos-s1/token_ids.txt` | 128 generated ids (space-separated) |
| `runs/s1-llamacpp-bf16-b11960/token_ids.txt` | exact reference ids |
| `/tmp/e2-myos-s1.stdout` | decode line + `CORRECT: PASS` |

## MyOS command

```bash
/workspace/MyOS/target/release/myos \
  --weights /workspace/MyOS/weights/Qwen2.5-1.5B-Instruct/model.safetensors \
  --prompt-ids /workspace/MyOS/arena/prompts/s1_prompt.token_ids.txt \
  --steps 128 \
  --ref-ids /workspace/MyOS/runs/s1-llamacpp-bf16-b11960/token_ids.txt \
  --fingerprint-dir /workspace/MyOS/runs/e2-myos-s1
```

- Prompt ids: `arena/prompts/s1_prompt.token_ids.txt` (128 ids; first id `151644`)
- Weights: SafeTensors BF16 mmap of Qwen2.5-1.5B-Instruct (~2.9G)
- Wall clock: ~11:50–12:21 UTC+8 (≈31 min); peak RSS observed ~3.0 GiB during run

## stdout (trimmed)

```
MyOS Phase1 weights decode (prompt=151644, steps=128): [785, 3015, 10775, …, 27865, 3015, 1526]
CORRECT: PASS
```

## Protocol status

- E1 preferred BF16 baseline: DET-PASS (see `arena/e1-s1-bf16-baseline.md`)
- E2 S1 exact vs that baseline: **CORRECT PASS** (this note)
- Side-track Q4 vs MyOS: **not opened**
- Next: optional S2/S3/S6; MyOS decode throughput optimization (still no_std / BF16 view / fp32 acc)
