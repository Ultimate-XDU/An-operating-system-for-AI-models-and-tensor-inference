//! Protocol §2.1 clocks and the S1 cost card.
//!
//! `t_load` is not part of `L_prefill` or `T_decode`. `t_prefill` ends before
//! the first new token. `t_decode` covers all `n_gen` new tokens. `M_peak` is
//! the max `VmRSS` sampled after load, through prefill and decode.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// llama.cpp S1 BF16 medians from `arena/e1-s1-bf16-cost-card.md` (`-t 8`).
pub const LLAMA_S1_BF16_RUN: &str = "s1-llamacpp-bf16-b11960";
pub const LLAMA_S1_BF16_T_DECODE: f64 = 17.2348;
pub const LLAMA_S1_BF16_L_TTFT_MS: f64 = 608.110;
pub const LLAMA_S1_BF16_L_TPOT_MS: f64 = 58.022;
pub const LLAMA_S1_BF16_M_PEAK_MIB: f64 = 3050.953;
pub const LLAMA_S1_BF16_THREADS: u64 = 8;

/// One measured process, before medians.
#[derive(Clone, Debug, PartialEq)]
pub struct PhaseSample {
    pub t_load_s: f64,
    pub t_prefill_s: f64,
    pub t_decode_s: f64,
    pub n_prompt: u64,
    pub n_gen: u64,
    pub m_peak_kib: Option<u64>,
    pub m_rss_after_load_kib: Option<u64>,
    /// `Some` when `--ref-ids` was compared. `None` when that file was absent.
    pub correct: Option<bool>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Derived {
    pub t_decode_tok_s: Option<f64>,
    pub l_ttft_ms: f64,
    pub l_prefill_ms: f64,
    pub l_tpot_ms: Option<f64>,
    pub t_e2e_tok_s: Option<f64>,
    pub m_peak_mib: Option<f64>,
    pub m_rss_after_load_mib: Option<f64>,
}

pub fn derive(sample: &PhaseSample) -> Derived {
    let l_ttft_ms = sample.t_prefill_s * 1000.0;
    let t_decode_tok_s = if sample.t_decode_s > 0.0 && sample.n_gen > 0 {
        Some(sample.n_gen as f64 / sample.t_decode_s)
    } else {
        None
    };
    let l_tpot_ms = if sample.n_gen > 0 {
        Some(sample.t_decode_s * 1000.0 / sample.n_gen as f64)
    } else {
        None
    };
    let t_e2e_tok_s = {
        let wall = sample.t_prefill_s + sample.t_decode_s;
        let tokens = sample.n_prompt + sample.n_gen;
        if wall > 0.0 && tokens > 0 {
            Some(tokens as f64 / wall)
        } else {
            None
        }
    };
    Derived {
        t_decode_tok_s,
        l_ttft_ms,
        l_prefill_ms: l_ttft_ms,
        l_tpot_ms,
        t_e2e_tok_s,
        m_peak_mib: sample.m_peak_kib.map(kib_to_mib),
        m_rss_after_load_mib: sample.m_rss_after_load_kib.map(kib_to_mib),
    }
}

fn kib_to_mib(kib: u64) -> f64 {
    kib as f64 / 1024.0
}

/// Odd counts: the middle value. Even counts: the mean of the two middles.
/// Same rule as Python `statistics.median`, which the llama S1 runner uses.
pub fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 1 {
        Some(sorted[n / 2])
    } else {
        Some((sorted[n / 2 - 1] + sorted[n / 2]) / 2.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Band {
    Win,
    Tie,
    Loss,
}

impl Band {
    pub fn as_str(self) -> &'static str {
        match self {
            Band::Win => "W",
            Band::Tie => "T",
            Band::Loss => "L",
        }
    }
}

/// Protocol §4.2. `higher_is_better` is `T_decode`. Latencies and `M_peak`
/// pass `false`. The ±5% edges are wins or losses (`>= 1.05` / `<= 0.95`).
pub fn band(myos: f64, base: f64, higher_is_better: bool) -> Option<Band> {
    if !myos.is_finite() || !base.is_finite() || base == 0.0 {
        return None;
    }
    let ratio = myos / base;
    Some(if higher_is_better {
        if ratio >= 1.05 {
            Band::Win
        } else if ratio <= 0.95 {
            Band::Loss
        } else {
            Band::Tie
        }
    } else if ratio <= 0.95 {
        Band::Win
    } else if ratio >= 1.05 {
        Band::Loss
    } else {
        Band::Tie
    })
}

fn band_points(band: Band, weight: i32) -> i32 {
    match band {
        Band::Win => weight,
        Band::Tie => 0,
        Band::Loss => -weight,
    }
}

/// §4.2 weights as integers: T_decode 35, L_TPOT 25, L_TTFT 20, M_peak 20.
/// `> 0` scene win, `< 0` scene loss, `0` scene tie.
pub fn scene_score(t_decode: Band, l_tpot: Band, l_ttft: Band, m_peak: Band) -> i32 {
    band_points(t_decode, 35)
        + band_points(l_tpot, 25)
        + band_points(l_ttft, 20)
        + band_points(m_peak, 20)
}

pub fn scene_band(score: i32) -> Band {
    if score > 0 {
        Band::Win
    } else if score < 0 {
        Band::Loss
    } else {
        Band::Tie
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SampleHeader {
    pub threads: u64,
    pub batch: u64,
    pub quant_or_dtype: String,
    pub acc_path: String,
    pub gemv_pack: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SampleFile {
    pub header: SampleHeader,
    pub sample: PhaseSample,
}

pub fn write_sample_file(dir: &Path, file: &SampleFile) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    let derived = derive(&file.sample);
    let mut out = String::new();
    out.push_str("{\n");
    push_str(&mut out, "scenario_id", "S1", true);
    push_u64(&mut out, "threads", file.header.threads, true);
    push_u64(&mut out, "batch", file.header.batch, true);
    push_str(
        &mut out,
        "quant_or_dtype",
        &file.header.quant_or_dtype,
        true,
    );
    push_str(&mut out, "acc_path", &file.header.acc_path, true);
    push_bool(&mut out, "gemv_pack", file.header.gemv_pack, true);
    push_u64(&mut out, "n_prompt", file.sample.n_prompt, true);
    push_u64(&mut out, "n_gen", file.sample.n_gen, true);
    push_f64(&mut out, "t_load_s", Some(file.sample.t_load_s), true);
    push_f64(&mut out, "t_prefill_s", Some(file.sample.t_prefill_s), true);
    push_f64(&mut out, "t_decode_s", Some(file.sample.t_decode_s), true);
    push_f64(&mut out, "T_decode_tok_s", derived.t_decode_tok_s, true);
    push_f64(&mut out, "L_TTFT_ms", Some(derived.l_ttft_ms), true);
    push_f64(&mut out, "L_prefill_ms", Some(derived.l_prefill_ms), true);
    push_f64(&mut out, "L_TPOT_ms", derived.l_tpot_ms, true);
    push_f64(&mut out, "T_e2e_tok_s", derived.t_e2e_tok_s, true);
    push_u64_opt(&mut out, "M_peak_KiB", file.sample.m_peak_kib, true);
    push_f64(&mut out, "M_peak_MiB", derived.m_peak_mib, true);
    push_u64_opt(
        &mut out,
        "M_rss_after_load_KiB",
        file.sample.m_rss_after_load_kib,
        true,
    );
    push_f64(
        &mut out,
        "M_rss_after_load_MiB",
        derived.m_rss_after_load_mib,
        true,
    );
    push_correct(&mut out, "correct", file.sample.correct, true);
    push_str(
        &mut out,
        "timing",
        "t_load is weight open+parse only and is outside L_prefill and T_decode. t_prefill is KV and workspace alloc plus prompt forwards, and it ends before the first new token. t_decode is greedy_decode of n_gen new tokens. M_peak is max VmRSS after load through the end of decode.",
        false,
    );
    out.push_str("}\n");
    fs::write(dir.join("metrics.json"), out)
}

pub fn read_sample_file(path: &Path) -> Result<SampleFile, String> {
    let text = fs::read_to_string(path).map_err(|err| format!("{}: {err}", path.display()))?;
    parse_sample_json(&text).map_err(|err| format!("{}: {err}", path.display()))
}

pub fn parse_sample_json(text: &str) -> Result<SampleFile, String> {
    let threads = json_u64(text, "threads")?;
    let batch = json_u64(text, "batch")?;
    let quant_or_dtype = json_string(text, "quant_or_dtype")?;
    let acc_path = json_string(text, "acc_path")?;
    let gemv_pack = json_bool(text, "gemv_pack")?;
    let n_prompt = json_u64(text, "n_prompt")?;
    let n_gen = json_u64(text, "n_gen")?;
    let t_load_s = json_f64_req(text, "t_load_s")?;
    let t_prefill_s = json_f64_req(text, "t_prefill_s")?;
    let t_decode_s = json_f64_req(text, "t_decode_s")?;
    let m_peak_kib = json_u64_opt(text, "M_peak_KiB")?;
    let m_rss_after_load_kib = json_u64_opt(text, "M_rss_after_load_KiB")?;
    let correct = json_correct(text, "correct")?;
    Ok(SampleFile {
        header: SampleHeader {
            threads,
            batch,
            quant_or_dtype,
            acc_path,
            gemv_pack,
        },
        sample: PhaseSample {
            t_load_s,
            t_prefill_s,
            t_decode_s,
            n_prompt,
            n_gen,
            m_peak_kib,
            m_rss_after_load_kib,
            correct,
        },
    })
}

#[derive(Clone, Debug)]
pub struct ShowdownMedians {
    pub t_load_s: f64,
    pub t_decode_tok_s: f64,
    pub l_ttft_ms: f64,
    pub l_tpot_ms: f64,
    pub t_e2e_tok_s: f64,
    pub m_peak_mib: f64,
}

pub fn medians_of(samples: &[PhaseSample]) -> Option<ShowdownMedians> {
    if samples.is_empty() {
        return None;
    }
    let derived: Vec<Derived> = samples.iter().map(derive).collect();
    Some(ShowdownMedians {
        t_load_s: median(&samples.iter().map(|s| s.t_load_s).collect::<Vec<_>>())?,
        t_decode_tok_s: median(
            &derived
                .iter()
                .filter_map(|d| d.t_decode_tok_s)
                .collect::<Vec<_>>(),
        )?,
        l_ttft_ms: median(&derived.iter().map(|d| d.l_ttft_ms).collect::<Vec<_>>())?,
        l_tpot_ms: median(
            &derived
                .iter()
                .filter_map(|d| d.l_tpot_ms)
                .collect::<Vec<_>>(),
        )?,
        t_e2e_tok_s: median(
            &derived
                .iter()
                .filter_map(|d| d.t_e2e_tok_s)
                .collect::<Vec<_>>(),
        )?,
        m_peak_mib: median(
            &derived
                .iter()
                .filter_map(|d| d.m_peak_mib)
                .collect::<Vec<_>>(),
        )?,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneLabel {
    /// §4.2 weighted sum.
    Scored(Band),
    /// Missing MyOS numbers, wrong `n_threads`, pack on, or batch != 1.
    Incomparable,
    /// CORRECT-FAIL or DET-FAIL. Throughput does not count (§4.2).
    GateLoss,
}

impl SceneLabel {
    pub fn as_str(self) -> &'static str {
        match self {
            SceneLabel::Scored(band) => band.as_str(),
            SceneLabel::Incomparable => "incomparable",
            SceneLabel::GateLoss => "L",
        }
    }
}

pub struct VerdictRow {
    pub ratio: Option<f64>,
    pub band: Option<Band>,
}

pub struct ShowdownVerdict {
    pub t_decode: VerdictRow,
    pub l_ttft: VerdictRow,
    pub l_tpot: VerdictRow,
    pub m_peak: VerdictRow,
    pub scene: SceneLabel,
    pub scene_score: Option<i32>,
}

/// `comparable` is false when threads, pack, or batch disagree with the
/// llama S1 BF16 fingerprint (`threads=8`, pack off, batch 1).
pub fn verdict(
    medians: Option<&ShowdownMedians>,
    comparable: bool,
    gates_pass: bool,
) -> ShowdownVerdict {
    let Some(medians) = medians else {
        return ShowdownVerdict {
            t_decode: VerdictRow {
                ratio: None,
                band: None,
            },
            l_ttft: VerdictRow {
                ratio: None,
                band: None,
            },
            l_tpot: VerdictRow {
                ratio: None,
                band: None,
            },
            m_peak: VerdictRow {
                ratio: None,
                band: None,
            },
            scene: SceneLabel::Incomparable,
            scene_score: None,
        };
    };
    let t_decode = row(medians.t_decode_tok_s, LLAMA_S1_BF16_T_DECODE, true);
    let l_ttft = row(medians.l_ttft_ms, LLAMA_S1_BF16_L_TTFT_MS, false);
    let l_tpot = row(medians.l_tpot_ms, LLAMA_S1_BF16_L_TPOT_MS, false);
    let m_peak = row(medians.m_peak_mib, LLAMA_S1_BF16_M_PEAK_MIB, false);
    if !comparable {
        return ShowdownVerdict {
            t_decode,
            l_ttft,
            l_tpot,
            m_peak,
            scene: SceneLabel::Incomparable,
            scene_score: None,
        };
    }
    if !gates_pass {
        return ShowdownVerdict {
            t_decode,
            l_ttft,
            l_tpot,
            m_peak,
            scene: SceneLabel::GateLoss,
            scene_score: None,
        };
    }
    match (t_decode.band, l_tpot.band, l_ttft.band, m_peak.band) {
        (Some(td), Some(tp), Some(tt), Some(mp)) => {
            let score = scene_score(td, tp, tt, mp);
            ShowdownVerdict {
                t_decode,
                l_ttft,
                l_tpot,
                m_peak,
                scene: SceneLabel::Scored(scene_band(score)),
                scene_score: Some(score),
            }
        }
        _ => ShowdownVerdict {
            t_decode,
            l_ttft,
            l_tpot,
            m_peak,
            scene: SceneLabel::Incomparable,
            scene_score: None,
        },
    }
}

fn row(myos: f64, base: f64, higher_is_better: bool) -> VerdictRow {
    VerdictRow {
        ratio: if base == 0.0 { None } else { Some(myos / base) },
        band: band(myos, base, higher_is_better),
    }
}

pub struct CostCard {
    /// Run directory name, for example `s1-myos-bf16-t8-k8`.
    pub fingerprint: String,
    pub threads: u64,
    pub batch: u64,
    pub gemv_pack: bool,
    pub n_prompt: u64,
    pub n_gen: u64,
    pub n_ctx: u64,
    pub warmup: usize,
    pub repeats: usize,
    pub commit: Option<String>,
    pub cmdline: String,
    pub correct: &'static str,
    pub det: &'static str,
    pub medians: Option<ShowdownMedians>,
    pub measured_on_this_machine: bool,
    pub extra_note: String,
}

pub fn render_cost_card(card: &CostCard) -> String {
    let comparable = card.threads == LLAMA_S1_BF16_THREADS && !card.gemv_pack && card.batch == 1;
    let gates_pass = card.correct == "PASS" && card.det == "PASS";
    let verdict = verdict(card.medians.as_ref(), comparable, gates_pass);
    let mut out = String::new();
    out.push_str(&format!(
        "# Cost Card · S1 · MyOS BF16 threads={}\n\n",
        card.threads
    ));
    out.push_str("### 设定\n");
    out.push_str(&format!("- fingerprint: `{}`\n", card.fingerprint));
    out.push_str("- model: Qwen2.5-1.5B-Instruct\n");
    out.push_str("- track: SafeTensors BF16 (main track)\n");
    out.push_str(&format!("- device: cpu, n_threads: {}\n", card.threads));
    out.push_str(&format!("- batch: {}\n", card.batch));
    out.push_str("- quant_or_dtype: bf16\n");
    out.push_str("- acc_path: bf16_w_fp32_acc\n");
    out.push_str(&format!("- gemv_pack: {}\n", card.gemv_pack));
    out.push_str("- deterministic: ON (greedy argmax). seed: null (no RNG)\n");
    out.push_str(&format!(
        "- n_ctx={}, n_prompt={}, n_gen={}\n",
        card.n_ctx, card.n_prompt, card.n_gen
    ));
    out.push_str(&format!(
        "- protocol: W={} discard, R={} median\n",
        card.warmup, card.repeats
    ));
    out.push_str(&format!(
        "- challenger commit: {}\n",
        card.commit.as_deref().unwrap_or("null")
    ));
    out.push_str(&format!(
        "- baseline: llama.cpp b11960 `{LLAMA_S1_BF16_RUN}` threads={LLAMA_S1_BF16_THREADS}\n"
    ));
    out.push_str(&format!(
        "- baseline source: `arena/e1-s1-bf16-cost-card.md` (T_decode {LLAMA_S1_BF16_T_DECODE}, L_TTFT {LLAMA_S1_BF16_L_TTFT_MS} ms, L_TPOT {LLAMA_S1_BF16_L_TPOT_MS} ms, M_peak {LLAMA_S1_BF16_M_PEAK_MIB} MiB)\n"
    ));
    out.push_str(&format!("- cmdline: `{}`\n", card.cmdline));
    if !card.measured_on_this_machine {
        out.push_str("- measurement: **UNMEASURED** on this machine (no 1.5B weight file in the authoring environment). The harness below is what the host runs.\n");
    }
    out.push_str("\n### 结果（中位数）\n");
    out.push_str("| metric | MyOS | llama.cpp | ratio (MyOS/base) | verdict |\n");
    out.push_str("|--------|------|-----------|-------------------|---------|\n");
    push_metric(
        &mut out,
        "T_decode (tok/s)",
        card.medians.as_ref().map(|m| m.t_decode_tok_s),
        LLAMA_S1_BF16_T_DECODE,
        &verdict.t_decode,
        4,
    );
    push_metric(
        &mut out,
        "L_TTFT / L_prefill (ms)",
        card.medians.as_ref().map(|m| m.l_ttft_ms),
        LLAMA_S1_BF16_L_TTFT_MS,
        &verdict.l_ttft,
        3,
    );
    push_metric(
        &mut out,
        "L_TPOT (ms)",
        card.medians.as_ref().map(|m| m.l_tpot_ms),
        LLAMA_S1_BF16_L_TPOT_MS,
        &verdict.l_tpot,
        3,
    );
    push_metric(
        &mut out,
        "M_peak (MiB RSS)",
        card.medians.as_ref().map(|m| m.m_peak_mib),
        LLAMA_S1_BF16_M_PEAK_MIB,
        &verdict.m_peak,
        3,
    );
    let load = card
        .medians
        .as_ref()
        .map(|m| format!("{:.4}", m.t_load_s))
        .unwrap_or_else(|| "UNMEASURED".to_string());
    out.push_str(&format!(
        "| t_load (s, not in T_decode) | {load} | — | — | appendix |\n"
    ));
    let e2e = card
        .medians
        .as_ref()
        .map(|m| format!("{:.4}", m.t_e2e_tok_s))
        .unwrap_or_else(|| "UNMEASURED".to_string());
    out.push_str(&format!("| T_e2e (tok/s) | {e2e} | — | — | appendix |\n"));
    out.push_str("\n### 正确性\n");
    out.push_str(&format!("- greedy exact match: {}\n", card.correct));
    out.push_str(&format!("- det consistency: {}\n", card.det));
    out.push_str("\n### 代价拆解\n");
    out.push_str(&format!(
        "- 算力：decode GEMV 按矩形切行。高而 K 短（gate/up/tied head）仍是 32 行块；长 K 与方阵（down、Q/K/V/O）走 8 行寄存器块（一段 YMM，每次 8 列）。分给 {} 个 worker。prefill 把整段 prompt 做成一次分块投影（同一权重行乘上 chunk 内每个 token）。每个 (token, 行) 仍是 `j=0..k-1` 先乘后加，非 FMA，无跨行归约，不 pack 权重。软件预取默认关（`MYOS_GEMV_PREFETCH=1` 才开）。刀⑩不拨预取/pack 开关。\n",
        card.threads
    ));
    out.push_str("- 内存：`M_peak` 是加载完成后、prefill 与 decode 期间的 VmRSS 峰值。权重保持 BF16 视图，无整表 f32。\n");
    out.push_str("- 工程：数值核 no_std；`std` 线程池只做确定行分割。刀⑥ panel pack 默认关（`gemv_pack` 见上）。\n");
    out.push_str("- 功能债：协议 batch 仍为 1。本卡不含 S2/S3，不含 Q4。\n");
    out.push_str("\n### 结论一句\n");
    out.push_str(&format!(
        "- 胜负：{}\n",
        scene_sentence(
            &verdict,
            comparable,
            gates_pass,
            card.measured_on_this_machine
        )
    ));
    out.push_str("- 主要代价：行维并行对齐 llama `-t 8`；不把未测数字写成 ±5% 胜负。\n");
    out.push_str("- 是否值得为 Phase2 继续加码：S1 闭环先落地。Phase2 与否看 host 填完本卡之后的场景分，不在本刀宣布。\n");
    if !card.extra_note.is_empty() {
        out.push_str(&format!("\n### 备注\n{}\n", card.extra_note));
    }
    out.push_str("\n### 计量边界（协议 §2.1）\n");
    out.push_str("1. `t_load` 单独计，不进入 L_prefill / T_decode。\n");
    out.push_str("2. `t_prefill`：从提交 prompt tokens 到产出第一个新 token 之前（含 KV / workspace 分配与整段 chunk prefill，不是 `n_prompt` 次 decode 同构 forward）。\n");
    out.push_str("3. `t_decode`：`greedy_decode` 跑完 `n_gen` 个新 token。`T_decode = n_gen / t_decode`，`L_TPOT = t_decode / n_gen`，`L_TTFT = t_prefill`。\n");
    out
}

fn scene_sentence(
    verdict: &ShowdownVerdict,
    comparable: bool,
    gates_pass: bool,
    measured: bool,
) -> String {
    if !measured {
        return "不可比。本机没有 1.5B 权重，MyOS 中位是 UNMEASURED。llama 分母已写在表里，host 重跑后按 §4.2 填 W/T/L。".to_string();
    }
    if !comparable {
        return "不可比（协议 §1.3）。threads、batch 或 gemv_pack 与 llama S1 BF16 `-t 8` / pack 关 / batch 1 不一致，不记 W/T/L。".to_string();
    }
    if !gates_pass {
        return "场景负。CORRECT 或 DET 未过，吞吐分作废（协议 §4.2）。".to_string();
    }
    match verdict.scene {
        SceneLabel::Scored(band) => format!(
            "场景{}（加权和 {:?}）。T_decode 0.35、L_TPOT 0.25、L_TTFT 0.20、M_peak 0.20；±5% 带宽。这不是「已经胜过 llama」的对外句，除非四项与场景分都是胜。",
            match band {
                Band::Win => "胜",
                Band::Tie => "平",
                Band::Loss => "负",
            },
            verdict.scene_score
        ),
        SceneLabel::Incomparable => "不可比。中位缺项，不能合成场景分。".to_string(),
        SceneLabel::GateLoss => "场景负。门闩未过。".to_string(),
    }
}

fn push_metric(
    out: &mut String,
    name: &str,
    myos: Option<f64>,
    base: f64,
    row: &VerdictRow,
    decimals: usize,
) {
    let mine = match myos {
        Some(v) => format_decimals(v, decimals),
        None => "UNMEASURED".to_string(),
    };
    let ratio = match row.ratio {
        Some(v) => format!("{v:.4}"),
        None => "—".to_string(),
    };
    let band = match row.band {
        Some(band) if myos.is_some() => band.as_str(),
        _ => "UNMEASURED",
    };
    out.push_str(&format!(
        "| {name} | {mine} | {} | {ratio} | {band} |\n",
        format_decimals(base, decimals)
    ));
}

fn format_decimals(value: f64, decimals: usize) -> String {
    format!("{:.*}", decimals, value)
}

pub fn ids_identical(groups: &[Vec<u32>]) -> bool {
    match groups.first() {
        Some(first) => groups.iter().all(|ids| ids == first),
        None => false,
    }
}

pub fn first_id_diff<'a>(
    groups: &'a [Vec<u32>],
) -> Option<(usize, usize, Option<u32>, Option<u32>)> {
    let first = groups.first()?;
    for (group_i, ids) in groups.iter().enumerate().skip(1) {
        let n = first.len().min(ids.len());
        for i in 0..n {
            if first[i] != ids[i] {
                return Some((group_i, i, Some(first[i]), Some(ids[i])));
            }
        }
        if first.len() != ids.len() {
            return Some((group_i, n, first.get(n).copied(), ids.get(n).copied()));
        }
    }
    None
}

pub fn parse_vm_rss_kib(text: &str) -> Option<u64> {
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            return rest.split_whitespace().next()?.parse().ok();
        }
    }
    None
}

pub fn vm_rss_kib() -> Option<u64> {
    let text = fs::read_to_string("/proc/self/status").ok()?;
    parse_vm_rss_kib(&text)
}

/// Samples `VmRSS` until [`RssSampler::stop`].
pub struct RssSampler {
    stop: Arc<AtomicBool>,
    peak: Arc<AtomicU64>,
    join: Option<JoinHandle<()>>,
}

impl RssSampler {
    pub fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(vm_rss_kib().unwrap_or(0)));
        let stop_bg = Arc::clone(&stop);
        let peak_bg = Arc::clone(&peak);
        let join = thread::Builder::new()
            .name("myos-rss".to_string())
            .spawn(move || {
                while !stop_bg.load(Ordering::Relaxed) {
                    if let Some(kib) = vm_rss_kib() {
                        peak_bg.fetch_max(kib, Ordering::Relaxed);
                    }
                    thread::sleep(Duration::from_millis(20));
                }
                if let Some(kib) = vm_rss_kib() {
                    peak_bg.fetch_max(kib, Ordering::Relaxed);
                }
            })
            .ok();
        Self { stop, peak, join }
    }

    pub fn stop(mut self) -> Option<u64> {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
        if let Some(kib) = vm_rss_kib() {
            self.peak.fetch_max(kib, Ordering::Relaxed);
        }
        let peak = self.peak.load(Ordering::Relaxed);
        if peak == 0 {
            None
        } else {
            Some(peak)
        }
    }
}

impl Drop for RssSampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

fn push_str(out: &mut String, key: &str, value: &str, comma: bool) {
    out.push_str("  \"");
    out.push_str(key);
    out.push_str("\": ");
    out.push_str(&json_escape(value));
    if comma {
        out.push(',');
    }
    out.push('\n');
}

fn push_u64(out: &mut String, key: &str, value: u64, comma: bool) {
    out.push_str(&format!("  \"{key}\": {value}"));
    if comma {
        out.push(',');
    }
    out.push('\n');
}

fn push_u64_opt(out: &mut String, key: &str, value: Option<u64>, comma: bool) {
    match value {
        Some(v) => push_u64(out, key, v, comma),
        None => {
            out.push_str(&format!("  \"{key}\": null"));
            if comma {
                out.push(',');
            }
            out.push('\n');
        }
    }
}

fn push_f64(out: &mut String, key: &str, value: Option<f64>, comma: bool) {
    out.push_str("  \"");
    out.push_str(key);
    out.push_str("\": ");
    match value {
        Some(v) if v.is_finite() => out.push_str(&format!("{v:.6}")),
        _ => out.push_str("null"),
    }
    if comma {
        out.push(',');
    }
    out.push('\n');
}

fn push_bool(out: &mut String, key: &str, value: bool, comma: bool) {
    out.push_str(&format!(
        "  \"{key}\": {}",
        if value { "true" } else { "false" }
    ));
    if comma {
        out.push(',');
    }
    out.push('\n');
}

fn push_correct(out: &mut String, key: &str, value: Option<bool>, comma: bool) {
    out.push_str("  \"");
    out.push_str(key);
    out.push_str("\": ");
    match value {
        Some(true) => out.push_str("\"PASS\""),
        Some(false) => out.push_str("\"FAIL\""),
        None => out.push_str("null"),
    }
    if comma {
        out.push(',');
    }
    out.push('\n');
}

fn json_escape(text: &str) -> String {
    let mut out = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_raw<'a>(text: &'a str, key: &str) -> Result<&'a str, String> {
    let pat = format!("\"{key}\": ");
    let Some(at) = text.find(&pat) else {
        return Err(format!("missing `{key}`"));
    };
    let rest = text[at + pat.len()..].trim_start();
    let end = rest
        .find([',', '\n', '}'])
        .ok_or_else(|| format!("unterminated `{key}`"))?;
    Ok(rest[..end].trim())
}

fn json_u64(text: &str, key: &str) -> Result<u64, String> {
    json_raw(text, key)?
        .parse()
        .map_err(|_| format!("`{key}` is not a u64"))
}

fn json_u64_opt(text: &str, key: &str) -> Result<Option<u64>, String> {
    let raw = json_raw(text, key)?;
    if raw == "null" {
        Ok(None)
    } else {
        raw.parse()
            .map(Some)
            .map_err(|_| format!("`{key}` is not a u64"))
    }
}

fn json_f64_req(text: &str, key: &str) -> Result<f64, String> {
    let raw = json_raw(text, key)?;
    if raw == "null" {
        return Err(format!("`{key}` is null"));
    }
    raw.parse().map_err(|_| format!("`{key}` is not a number"))
}

fn json_bool(text: &str, key: &str) -> Result<bool, String> {
    match json_raw(text, key)? {
        "true" => Ok(true),
        "false" => Ok(false),
        other => Err(format!("`{key}` is `{other}`, want true or false")),
    }
}

fn json_string(text: &str, key: &str) -> Result<String, String> {
    let raw = json_raw(text, key)?;
    if raw == "null" {
        return Err(format!("`{key}` is null"));
    }
    let raw = raw
        .strip_prefix('"')
        .ok_or_else(|| format!("`{key}` is not a string"))?;
    let raw = raw
        .strip_suffix('"')
        .ok_or_else(|| format!("`{key}` is not a string"))?;
    Ok(raw.to_string())
}

fn json_correct(text: &str, key: &str) -> Result<Option<bool>, String> {
    match json_raw(text, key)? {
        "null" => Ok(None),
        "\"PASS\"" => Ok(Some(true)),
        "\"FAIL\"" => Ok(Some(false)),
        other => Err(format!("`{key}` is `{other}`")),
    }
}

/// Write `path`, creating parent directories.
pub fn write_text(path: &Path, text: &str) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    file.write_all(text.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(t_decode_s: f64, peak_kib: u64) -> PhaseSample {
        PhaseSample {
            t_load_s: 1.5,
            t_prefill_s: 0.608110,
            t_decode_s,
            n_prompt: 128,
            n_gen: 128,
            m_peak_kib: Some(peak_kib),
            m_rss_after_load_kib: Some(peak_kib / 2),
            correct: Some(true),
        }
    }

    #[test]
    fn median_matches_the_middle_and_the_even_mean() {
        assert_eq!(median(&[]), None);
        assert_eq!(median(&[3.0]), Some(3.0));
        assert_eq!(median(&[5.0, 1.0, 4.0]), Some(4.0));
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), Some(2.5));
    }

    #[test]
    fn bands_use_the_five_percent_edges() {
        assert_eq!(band(17.2348 * 1.05, 17.2348, true), Some(Band::Win));
        assert_eq!(band(17.2348 * 1.049, 17.2348, true), Some(Band::Tie));
        assert_eq!(band(17.2348, 17.2348, true), Some(Band::Tie));
        assert_eq!(band(17.2348 * 0.95, 17.2348, true), Some(Band::Loss));
        assert_eq!(band(58.022 * 0.95, 58.022, false), Some(Band::Win));
        assert_eq!(band(58.022 * 1.05, 58.022, false), Some(Band::Loss));
        assert_eq!(band(58.022, 58.022, false), Some(Band::Tie));
    }

    #[test]
    fn scene_score_uses_integer_weights() {
        assert_eq!(scene_score(Band::Tie, Band::Tie, Band::Tie, Band::Tie), 0);
        assert_eq!(scene_band(0), Band::Tie);
        // T_decode loss, the other three wins: -35+25+20+20 = 30.
        let score = scene_score(Band::Loss, Band::Win, Band::Win, Band::Win);
        assert_eq!(score, 30);
        assert_eq!(scene_band(score), Band::Win);
        let loss = scene_score(Band::Loss, Band::Loss, Band::Win, Band::Win);
        assert_eq!(loss, -20);
        assert_eq!(scene_band(loss), Band::Loss);
    }

    #[test]
    fn derive_splits_prefill_and_decode() {
        let d = derive(&sample(7.426816, 3_124_176));
        let expect_tok_s = 128.0 / 7.426816;
        assert!((d.t_decode_tok_s.unwrap() - expect_tok_s).abs() < 1e-9);
        assert!((d.l_ttft_ms - 608.110).abs() < 1e-6);
        assert!((d.l_prefill_ms - d.l_ttft_ms).abs() < 1e-12);
        assert!((d.l_tpot_ms.unwrap() - 7.426816 * 1000.0 / 128.0).abs() < 1e-9);
        assert!((d.m_peak_mib.unwrap() - 3124176.0 / 1024.0).abs() < 1e-9);
        // t_load is not inside T_e2e.
        let e2e = 256.0 / (0.608110 + 7.426816);
        assert!((d.t_e2e_tok_s.unwrap() - e2e).abs() < 1e-9);
    }

    #[test]
    fn sample_json_round_trips_and_keeps_null_peak() {
        let file = SampleFile {
            header: SampleHeader {
                threads: 8,
                batch: 1,
                quant_or_dtype: "bf16".to_string(),
                acc_path: "bf16_w_fp32_acc".to_string(),
                gemv_pack: false,
            },
            sample: PhaseSample {
                m_peak_kib: None,
                m_rss_after_load_kib: None,
                correct: None,
                ..sample(4.0, 0)
            },
        };
        let dir = std::env::temp_dir().join(format!(
            "myos-meter-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        write_sample_file(&dir, &file).unwrap();
        let parsed = read_sample_file(&dir.join("metrics.json")).unwrap();
        assert_eq!(parsed.header.threads, 8);
        assert_eq!(parsed.header.quant_or_dtype, "bf16");
        assert_eq!(parsed.header.acc_path, "bf16_w_fp32_acc");
        assert!(!parsed.header.gemv_pack);
        assert_eq!(parsed.header.batch, 1);
        assert_eq!(parsed.sample.m_peak_kib, None);
        assert_eq!(parsed.sample.correct, None);
        assert!((parsed.sample.t_decode_s - 4.0).abs() < 1e-5);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn vm_rss_line_parses_kib() {
        let text = "Name:\tmyos\nVmRSS:\t  3050976 kB\nVmSize:\t1 kB\n";
        assert_eq!(parse_vm_rss_kib(text), Some(3_050_976));
        assert_eq!(parse_vm_rss_kib("no rss\n"), None);
    }

    #[test]
    fn unmeasured_card_keeps_llama_denominator_and_refuses_a_win() {
        let text = render_cost_card(&CostCard {
            fingerprint: "s1-myos-bf16-t8-k8".to_string(),
            threads: 8,
            batch: 1,
            gemv_pack: false,
            n_prompt: 128,
            n_gen: 128,
            n_ctx: 256,
            warmup: 3,
            repeats: 5,
            commit: None,
            cmdline: "myos --arena-s1".to_string(),
            correct: "UNMEASURED",
            det: "UNMEASURED",
            medians: None,
            measured_on_this_machine: false,
            extra_note: String::new(),
        });
        assert!(text.contains("s1-myos-bf16-t8-k8"), "{text}");
        assert!(text.contains("分块投影"), "{text}");
        assert!(text.contains("UNMEASURED"), "{text}");
        assert!(text.contains("17.2348"), "{text}");
        assert!(text.contains("608.110"), "{text}");
        assert!(text.contains("58.022"), "{text}");
        assert!(text.contains("3050.953"), "{text}");
        assert!(text.contains("bf16_w_fp32_acc"), "{text}");
        assert!(text.contains("gemv_pack: false"), "{text}");
        assert!(text.contains("n_threads: 8"), "{text}");
        assert!(text.contains("不可比"), "{text}");
        assert!(!text.contains("| W |"), "{text}");
    }

    #[test]
    fn slower_showdown_is_a_loss_not_a_missing_cell() {
        let medians = ShowdownMedians {
            t_load_s: 2.0,
            t_decode_tok_s: 2.0,
            l_ttft_ms: 8000.0,
            l_tpot_ms: 500.0,
            t_e2e_tok_s: 1.0,
            m_peak_mib: 4000.0,
        };
        let text = render_cost_card(&CostCard {
            fingerprint: "s1-myos-bf16-t8-k8".to_string(),
            threads: 8,
            batch: 1,
            gemv_pack: false,
            n_prompt: 128,
            n_gen: 128,
            n_ctx: 256,
            warmup: 3,
            repeats: 5,
            commit: Some("abc".to_string()),
            cmdline: "myos".to_string(),
            correct: "PASS",
            det: "PASS",
            medians: Some(medians),
            measured_on_this_machine: true,
            extra_note: String::new(),
        });
        assert!(text.contains("| L |"), "{text}");
        assert!(text.contains("场景负"), "{text}");
    }

    #[test]
    fn threads_1_is_incomparable_even_with_numbers() {
        let verdict = verdict(
            Some(&ShowdownMedians {
                t_load_s: 1.0,
                t_decode_tok_s: 100.0,
                l_ttft_ms: 1.0,
                l_tpot_ms: 1.0,
                t_e2e_tok_s: 100.0,
                m_peak_mib: 1.0,
            }),
            false,
            true,
        );
        assert_eq!(verdict.scene, SceneLabel::Incomparable);
        assert!(verdict.t_decode.ratio.unwrap() > 1.0);
    }
}
