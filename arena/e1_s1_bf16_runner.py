#!/usr/bin/env python3
"""E1 main-track S1 runner for self-converted BF16 GGUF (b11960). Preferred for MyOS ST-BF16 exact."""
from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

# --- config (overridable via env) ---
BIN = os.environ.get("E1_BIN", "/workspace/tools/llama.cpp-bin/llama-b11960/llama-cli")
LIB = os.environ.get("E1_LIB", "/workspace/tools/llama.cpp-bin/llama-b11960")
GGUF = "/workspace/MyOS/weights/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-bf16.gguf"
PROMPT = "/workspace/MyOS/arena/prompts/s1_prompt.txt"
MODEL_SHA = "1a32a39cefefe73f82b4600ccecfe4f3f9411337543a7481f1aa3330d71c973b"
COMMIT = "8642541b255bfb5dc39cd351a178be09013a55d6"
THREADS = int(os.environ.get("E1_THREADS", "8"))
N_CTX = 512
N_GEN = 128
W = 3
R = 5
RAW = Path(os.environ.get("E1_RAW", "/tmp/e1-s1-bf16-raw"))
RAW.mkdir(parents=True, exist_ok=True)
FINGERPRINT_ID = os.environ.get("E1_FINGERPRINT_ID", "s1-llamacpp-bf16-b11960")
RUN_DIR = Path("/workspace/MyOS/runs") / FINGERPRINT_ID
TRACK = "main-track-gguf-bf16"
EXTRA_ENV = {}
# DET-stable extras when using noamx
if os.environ.get("E1_DET_STABLE") == "1":
    EXTRA_ENV = {
        "OMP_NUM_THREADS": "1",
        "GGML_CPU_TILED_MM": "0",
        "GGML_CPU_DISABLE_FUSION": "1",
    }

CMD = [
    BIN,
    "-m", GGUF,
    "-ngl", "0",
    "-t", str(THREADS),
    "-c", str(N_CTX),
    "-n", str(N_GEN),
    "--temp", "0",
    "--top-k", "1",
    "--top-p", "1.0",
    "--min-p", "0.0",
    "--typical", "1.0",
    "--repeat-penalty", "1.0",
    "--presence-penalty", "0.0",
    "--frequency-penalty", "0.0",
    "--dry-multiplier", "0.0",
    "-s", "0",
    "-f", PROMPT,
    "--no-display-prompt",
    "--single-turn",
    "--no-warmup",
    "-v",
]
CMDLINE = " ".join(CMD)

NEXT_TOKEN_RE = re.compile(
    r"next token:\s*(-?\d+)\s+'((?:\\'|\\\\|[^'])*)'"
)


def poll_rss(pid: int, peak: list[int], stop: threading.Event):
    while not stop.is_set():
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        kb = int(line.split()[1])
                        if kb > peak[0]:
                            peak[0] = kb
                        break
        except (FileNotFoundError, ProcessLookupError, ValueError):
            break
        time.sleep(0.05)


def parse_token_ids(stderr: str) -> list[int]:
    ids = []
    for m in NEXT_TOKEN_RE.finditer(stderr):
        ids.append(int(m.group(1)))
    return ids


def parse_timings(stderr: str) -> dict:
    out = {}
    m = re.search(
        r"prompt eval time =\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens\s*\(\s*([0-9.]+)\s*ms per token,\s*([0-9.]+)\s*tokens per second\)",
        stderr,
    )
    if m:
        out["prompt_ms"] = float(m.group(1))
        out["prompt_n"] = int(m.group(2))
        out["prompt_ms_per_tok"] = float(m.group(3))
        out["prompt_tok_s"] = float(m.group(4))
    m = re.search(
        r"(?m)^[^\n]*?(?<!prompt )eval time =\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens\s*\(\s*([0-9.]+)\s*ms per token,\s*([0-9.]+)\s*tokens per second\)",
        stderr,
    )
    if not m:
        m = re.search(
            r"(?m)^\s*eval time =\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens\s*\(\s*([0-9.]+)\s*ms per token,\s*([0-9.]+)\s*tokens per second\)",
            stderr,
        )
    if m:
        out["eval_ms"] = float(m.group(1))
        out["eval_n"] = int(m.group(2))
        out["eval_ms_per_tok"] = float(m.group(3))
        out["eval_tok_s"] = float(m.group(4))
    m = re.search(r'Parsed message: \{"role":"assistant","content":"(.*?)"\}', stderr)
    if m:
        try:
            out["gen_text"] = json.loads('"' + m.group(1) + '"')
        except json.JSONDecodeError:
            out["gen_text"] = m.group(1).encode("utf-8").decode("unicode_escape")
    return out


def extract_stdout_gen(stdout: str) -> str:
    lines = stdout.splitlines()
    end = None
    for i, ln in enumerate(lines):
        if ln.startswith("[ Prompt:"):
            end = i
    if end is None:
        return ""
    start = None
    for i, ln in enumerate(lines):
        if ln.startswith("> "):
            start = i
            break
    if start is None:
        return ""
    i = start + 1
    while i < end and lines[i].strip() != "":
        if "(truncated)" in lines[i] or lines[i].startswith("<|"):
            i += 1
            continue
        break
    while i < end and lines[i].strip() == "":
        i += 1
    gen_lines = []
    while i < end:
        gen_lines.append(lines[i])
        i += 1
    return "\n".join(gen_lines).rstrip("\n")


def run_once(tag: str) -> dict:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LIB + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env.update(EXTRA_ENV)
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    peak = [0]
    stop = threading.Event()
    th = threading.Thread(target=poll_rss, args=(proc.pid, peak, stop), daemon=True)
    th.start()
    stdout, stderr = proc.communicate()
    stop.set()
    th.join(timeout=1)
    wall_s = time.perf_counter() - t0
    (RAW / f"{tag}.stdout.txt").write_text(stdout)
    (RAW / f"{tag}.stderr.txt").write_text(stderr)
    timings = parse_timings(stderr)
    token_ids = parse_token_ids(stderr)
    (RAW / f"{tag}.token_ids.txt").write_text(" ".join(str(x) for x in token_ids) + ("\n" if token_ids else ""))
    gen = timings.get("gen_text") or extract_stdout_gen(stdout)
    result = {
        "tag": tag,
        "exit_code": proc.returncode,
        "wall_s": wall_s,
        "peak_rss_kib": peak[0],
        "peak_rss_mib": round(peak[0] / 1024.0, 3) if peak[0] else None,
        "prompt_ms": timings.get("prompt_ms"),
        "prompt_n": timings.get("prompt_n"),
        "prompt_tok_s": timings.get("prompt_tok_s"),
        "eval_ms": timings.get("eval_ms"),
        "eval_n": timings.get("eval_n"),
        "eval_tok_s": timings.get("eval_tok_s"),
        "eval_ms_per_tok": timings.get("eval_ms_per_tok"),
        "gen_text": gen,
        "token_ids_path": str(RAW / f"{tag}.token_ids.txt"),
        "n_token_ids": len(token_ids),
        "token_ids": token_ids,
        "bin": BIN,
        "lib": LIB,
        "threads": THREADS,
        "extra_env": dict(EXTRA_ENV),
    }
    if result["prompt_ms"] is not None:
        result["L_TTFT_ms"] = result["prompt_ms"]
    if result["eval_ms"] is not None and result["eval_n"]:
        result["L_TPOT_ms"] = result["eval_ms"] / result["eval_n"]
        result["T_decode_tok_s"] = result["eval_n"] / (result["eval_ms"] / 1000.0)
    if result["prompt_ms"] is not None and result["eval_ms"] is not None and result["prompt_n"] and result["eval_n"]:
        tot_ms = result["prompt_ms"] + result["eval_ms"]
        result["T_e2e_tok_s"] = (result["prompt_n"] + result["eval_n"]) / (tot_ms / 1000.0)
    slim = {k: v for k, v in result.items() if k != "token_ids"}
    (RAW / f"{tag}.result.json").write_text(json.dumps(slim, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "tag",
                    "exit_code",
                    "prompt_n",
                    "eval_n",
                    "prompt_ms",
                    "eval_ms",
                    "T_decode_tok_s",
                    "peak_rss_mib",
                    "n_token_ids",
                )
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    print("GEN_PREFIX:", (gen or "")[:80].replace("\n", "\\n"), flush=True)
    return result


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def first_diff(ids_lists):
    if not ids_lists:
        return None
    ref = ids_lists[0]
    for si, ids in enumerate(ids_lists[1:], start=2):
        n = min(len(ref), len(ids))
        for i in range(n):
            if ref[i] != ids[i]:
                return {
                    "vs": f"score{si}",
                    "index": i,
                    "ref_id": ref[i],
                    "other_id": ids[i],
                    "ref_len": len(ref),
                    "other_len": len(ids),
                }
        if len(ref) != len(ids):
            return {
                "vs": f"score{si}",
                "index": n,
                "ref_id": None if n >= len(ref) else ref[n],
                "other_id": None if n >= len(ids) else ids[n],
                "ref_len": len(ref),
                "other_len": len(ids),
                "note": "length mismatch",
            }
    return None


def main():
    print("CMDLINE:", CMDLINE, flush=True)
    print("THREADS:", THREADS, flush=True)
    print("BIN:", BIN, flush=True)
    print("LIB:", LIB, flush=True)
    print("GGUF_SHA:", MODEL_SHA, flush=True)
    print("EXTRA_ENV:", EXTRA_ENV, flush=True)
    print("FINGERPRINT_ID:", FINGERPRINT_ID, flush=True)
    warm = []
    for i in range(1, W + 1):
        print(f"=== WARM {i}/{W} ===", flush=True)
        warm.append(run_once(f"warm{i}"))
    scored = []
    for i in range(1, R + 1):
        print(f"=== SCORE {i}/{R} ===", flush=True)
        scored.append(run_once(f"score{i}"))

    texts = [r["gen_text"] for r in scored]
    ids_lists = [r["token_ids"] for r in scored]
    text_identical = all(t == texts[0] and t is not None and t != "" for t in texts)
    ids_identical = (
        len(ids_lists) == R
        and all(len(x) == N_GEN for x in ids_lists)
        and all(x == ids_lists[0] for x in ids_lists)
    )
    gen_ns = [r.get("eval_n") for r in scored]
    fd = first_diff(ids_lists)
    det_pass = ids_identical and all(n == N_GEN for n in gen_ns)
    if det_pass:
        det_note = f"DET-PASS ids_identical=True text_identical={text_identical} lens={[len(x) for x in ids_lists]}"
    else:
        fd_s = ""
        if fd:
            fd_s = f"; first_diff score1 vs {fd['vs']}@{fd['index']}: {fd['ref_id']} vs {fd['other_id']}"
        det_note = (
            f"DET-FAIL ids_identical={ids_identical} text_identical={text_identical} "
            f"lens={[len(x) for x in ids_lists]}{fd_s}"
        )

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    n_prompt = scored[0].get("prompt_n") or None
    prompt_ns = [r.get("prompt_n") for r in scored]

    bin_note = BIN
    if "noamx" in BIN:
        bin_note += " (DET-stable: no sapphirerapids/AMX; -t1 + OMP/GGML fusion clamps)"

    fp = {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_sha256": MODEL_SHA,
        "fmt": "gguf",
        "quant_or_dtype": "bf16",
        "n_ctx": N_CTX,
        "n_prompt": n_prompt,
        "n_gen": N_GEN,
        "batch": 1,
        "threads": THREADS,
        "device": "cpu",
        "deterministic": True,
        "seed": 0,
        "commit": COMMIT,
        "cmdline": CMDLINE,
        "scenario_id": "S1",
        "run_id": FINGERPRINT_ID,
        "track": TRACK,
        "impl": "llama.cpp",
        "llama_build": "b11960" if "noamx" not in BIN else "b11960-noamx",
        "llama_bin": BIN,
        "llama_lib": LIB,
        "extra_env": dict(EXTRA_ENV),
        "prompt_file": PROMPT,
        "sampler_extras": "--top-p 1.0 --min-p 0.0 --typical 1.0 --presence-penalty 0 --frequency-penalty 0 --dry-multiplier 0",
        "gguf_source": "self-converted via convert_hf_to_gguf.py --outtype bf16 from /workspace/MyOS/weights/Qwen2.5-1.5B-Instruct",
        "notes": (
            "MAIN TRACK self-converted BF16 GGUF — PREFERRED llama baseline for later E2 exact vs MyOS ST-BF16. "
            "Not Q4 side-track; not official FP16. "
            "Prompt file = user body only; CLI default Qwen jinja adds system "
            "'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' "
            f"Measured n_prompt values={prompt_ns}. W={W} discard R={R} median. "
            f"Binary: {bin_note}. "
            f"{det_note}. "
            + (
                "Per 博弈手: DET-FAIL ⇒ 吞吐/延迟/内存参考 · 胜负与 CORRECT 作废; E1 main-track not closed for this fingerprint."
                if not det_pass
                else "DET-PASS: usable main-track E1 BF16 baseline for later E2 exact vs MyOS ST-BF16."
            )
        ),
    }
    (RUN_DIR / "run_fingerprint.json").write_text(json.dumps(fp, indent=2) + "\n")

    (RUN_DIR / "tokens.txt").write_text((texts[0] or "") + "\n")
    (RUN_DIR / "token_ids.txt").write_text(
        " ".join(str(x) for x in ids_lists[0]) + ("\n" if ids_lists[0] else "")
    )

    def slim_run(r):
        return {k: v for k, v in r.items() if k != "token_ids"}

    metrics = {
        "scenario_id": "S1",
        "track": TRACK,
        "impl": "llama.cpp",
        "quant_or_dtype": "bf16",
        "W": W,
        "R": R,
        "det": "PASS" if det_pass else "FAIL",
        "det_note": det_note,
        "first_diff": fd,
        "binary": BIN,
        "threads": THREADS,
        "extra_env": dict(EXTRA_ENV),
        "median": {
            "L_TTFT_ms": median([r.get("L_TTFT_ms") for r in scored]),
            "L_TPOT_ms": median([r.get("L_TPOT_ms") for r in scored]),
            "T_decode_tok_s": median([r.get("T_decode_tok_s") for r in scored]),
            "T_e2e_tok_s": median([r.get("T_e2e_tok_s") for r in scored]),
            "M_peak_MiB": median([r.get("peak_rss_mib") for r in scored]),
            "prompt_ms": median([r.get("prompt_ms") for r in scored]),
            "eval_ms": median([r.get("eval_ms") for r in scored]),
            "wall_s": median([r.get("wall_s") for r in scored]),
        },
        "per_run_scored": [slim_run(r) for r in scored],
        "per_run_warmup_discarded": [slim_run(r) for r in warm],
        "collected_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "timezone_note": "Etc/GMT-8 (UTC+8)",
        "CORRECT": None if not det_pass else "N/A_pending_E2",
        "verdict_note": (
            "吞吐/延迟/内存参考 · 胜负与 CORRECT 作废"
            if not det_pass
            else "DET-PASS medians usable as preferred main-track BF16 baseline; CORRECT vs MyOS deferred to E2"
        ),
    }
    (RUN_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")

    det_ev = {
        "det": det_note,
        "first_diff": fd,
        "ids_identical": ids_identical,
        "text_identical": text_identical,
        "scored_texts": texts,
        "scored_token_ids": ids_lists,
        "token_id_source": "llama-cli -v stderr 'next token: <id>' lines",
        "binary": BIN,
        "threads": THREADS,
        "extra_env": dict(EXTRA_ENV),
    }
    (RUN_DIR / "det_evidence.json").write_text(json.dumps(det_ev, indent=2, ensure_ascii=False) + "\n")

    med = metrics["median"]
    det_label = "PASS" if det_pass else "FAIL"
    ref_label = (
        "吞吐/延迟/内存参考 · 胜负与 CORRECT 作废"
        if not det_pass
        else "usable preferred BF16 main-track baseline (E2 exact vs MyOS ST-BF16)"
    )
    card = f"""# Cost Card · S1 · {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')} (UTC+8)

### 设定
- fingerprint: `{FINGERPRINT_ID}`
- model: Qwen2.5-1.5B-Instruct
- track: **self-converted GGUF BF16 (MAIN TRACK · PREFERRED for MyOS exact)**
- device: cpu, n_threads: {THREADS}
- baseline: llama.cpp {fp['llama_build']} / commit `{COMMIT}`
- binary: `{BIN}`
- MyOS challenger: **not run** (llama-only E1 baseline)
- prompt: `{PROMPT}` (user body; CLI Qwen chat template → measured n_prompt={n_prompt})
- n_ctx={N_CTX}, n_gen={N_GEN}, batch=1, deterministic=ON, seed=0
- protocol: W={W} discard, R={R} median
- GGUF sha256: `{MODEL_SHA}` (convert_hf_to_gguf.py --outtype bf16)
- extra_env: `{json.dumps(EXTRA_ENV)}`

### Exact cmdline
```
LD_LIBRARY_PATH={LIB} \\
{(" ".join(f"{k}={v}" for k,v in EXTRA_ENV.items()) + " \\" ) if EXTRA_ENV else ""}
{CMDLINE}
```

### 结果（中位数，llama.cpp only）
| metric | llama.cpp | MyOS | ratio | verdict |
|--------|-----------|------|-------|---------|
| T_decode (tok/s) | {med['T_decode_tok_s']:.4f} | — | — | {ref_label} |
| L_TTFT / L_prefill (ms) | {med['L_TTFT_ms']:.3f} | — | — | {ref_label} |
| L_TPOT (ms) | {med['L_TPOT_ms']:.3f} | — | — | {ref_label} |
| M_peak (MiB RSS) | {med['M_peak_MiB']:.3f} | — | — | {ref_label} |
| T_e2e (tok/s) | {med['T_e2e_tok_s']:.4f} | — | — | appendix |

### DET
- **{det_label}**: {det_note}
- Evidence: `det_evidence.json`, per-run ids under `{RAW}/`
- CORRECT: {"N/A (DET-FAIL)" if not det_pass else "N/A pending E2 vs MyOS ST-BF16"}

### Generated text (score1; common only if DET-PASS)
```
{texts[0]}
```

### Notes
- Preferred main-track BF16 GGUF for later E2 exact vs MyOS ST-BF16 (same dtype family).
- Timings from llama.cpp `slot print_timing` (prompt eval / eval); M_peak = max VmRSS sampled from `/proc/<pid>/status`.
- CLI internal empty warmup off (`--no-warmup`); protocol warm-ups are full discarded runs.
- Token ids parsed from `-v` stderr `next token:` lines (same method as FP16/Q4 runs).
{"- Per 博弈手 §4.2: DET-FAIL ⇒ 场景吞吐胜负作废; medians kept as 参考 only; E1 main-track not closed for this fingerprint." if not det_pass else "- DET-PASS: E1 main-track BF16 baseline closed for later E2 exact vs MyOS."}
"""
    (RUN_DIR / "cost-card.md").write_text(card)
    print("RUN_DIR", RUN_DIR, flush=True)
    print("DET", det_note, flush=True)
    print("MEDIAN", json.dumps(med), flush=True)
    print("FIRST_DIFF", json.dumps(fd), flush=True)
    print("N_PROMPT", n_prompt, "all", prompt_ns, flush=True)
    print("DET_PASS", det_pass, flush=True)


if __name__ == "__main__":
    main()
