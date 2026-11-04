//! S1 showdown runner: W discarded processes, then R scored processes.
//!
//! Each child is a fresh `myos --weights` decode so RSS and `t_load` match
//! a cold process. This parent does not decode. It writes `metrics.json`,
//! `det_evidence.json`, `tokens.txt`, `token_ids.txt`, and the cost card.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use crate::fingerprint::write_token_ids;
use crate::meter::{
    self, ids_identical, read_sample_file, render_cost_card, write_text, CostCard, PhaseSample,
    SampleFile, SceneLabel, ShowdownMedians, LLAMA_S1_BF16_THREADS,
};

pub struct ArenaJob {
    pub bin: PathBuf,
    pub weights: String,
    pub prompt_ids: String,
    pub ref_ids: String,
    pub steps: usize,
    pub threads: usize,
    pub warmup: usize,
    pub repeats: usize,
    pub out_dir: PathBuf,
    pub cost_card: PathBuf,
    pub cmdline: String,
    pub commit: Option<String>,
}

pub struct ArenaReport {
    pub det_pass: bool,
    pub correct_pass: bool,
    pub comparable: bool,
    pub scene: SceneLabel,
    pub medians: ShowdownMedians,
}

pub fn run(job: &ArenaJob) -> Result<ArenaReport, String> {
    if job.repeats == 0 {
        return Err("--repeats must be at least 1".to_string());
    }
    fs::create_dir_all(&job.out_dir).map_err(|err| format!("{}: {err}", job.out_dir.display()))?;
    let reference = read_ids(Path::new(&job.ref_ids))?;
    let mut warmup_samples = Vec::with_capacity(job.warmup);
    let mut warmup_ids = Vec::with_capacity(job.warmup);
    let mut gemv_pack = None;
    for i in 0..job.warmup {
        eprintln!("=== WARM {}/{} ===", i + 1, job.warmup);
        let child = run_child(job, &job.out_dir.join(format!("warm-{}", i + 1)))?;
        gemv_pack = Some(same_pack(gemv_pack, child.gemv_pack)?);
        warmup_samples.push(child.sample);
        warmup_ids.push(child.ids);
    }
    let mut scored_samples = Vec::with_capacity(job.repeats);
    let mut scored_ids = Vec::with_capacity(job.repeats);
    for i in 0..job.repeats {
        eprintln!("=== SCORE {}/{} ===", i + 1, job.repeats);
        let child = run_child(job, &job.out_dir.join(format!("score-{}", i + 1)))?;
        gemv_pack = Some(same_pack(gemv_pack, child.gemv_pack)?);
        scored_samples.push(child.sample);
        scored_ids.push(child.ids);
    }
    summarize(
        job,
        gemv_pack.unwrap_or(false),
        &reference,
        &warmup_samples,
        &warmup_ids,
        &scored_samples,
        &scored_ids,
    )
}

fn same_pack(seen: Option<bool>, next: bool) -> Result<bool, String> {
    if let Some(seen) = seen {
        if seen != next {
            return Err("children disagree on gemv_pack".to_string());
        }
    }
    Ok(next)
}

struct ChildRun {
    sample: PhaseSample,
    ids: Vec<u32>,
    gemv_pack: bool,
}

fn run_child(job: &ArenaJob, dir: &Path) -> Result<ChildRun, String> {
    fs::create_dir_all(dir).map_err(|err| format!("{}: {err}", dir.display()))?;
    let mut cmd = Command::new(&job.bin);
    cmd.arg("--weights")
        .arg(&job.weights)
        .arg("--prompt-ids")
        .arg(&job.prompt_ids)
        .arg("--ref-ids")
        .arg(&job.ref_ids)
        .arg("--steps")
        .arg(job.steps.to_string())
        .arg("--threads")
        .arg(job.threads.to_string())
        .arg("--fingerprint-dir")
        .arg(dir);
    let started = Instant::now();
    let output = cmd
        .output()
        .map_err(|err| format!("spawn {}: {err}", job.bin.display()))?;
    let wall_s = started.elapsed().as_secs_f64();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let _ = fs::write(dir.join("stdout.txt"), stdout.as_bytes());
    let _ = fs::write(dir.join("stderr.txt"), stderr.as_bytes());
    let _ = fs::write(dir.join("wall_s.txt"), format!("{wall_s:.6}\n"));
    if !output.status.success() {
        return Err(format!(
            "child {} exited {}: {}",
            dir.display(),
            output.status.code().unwrap_or(1),
            stderr_tail(&stderr)
        ));
    }
    let file = read_sample_file(&dir.join("metrics.json"))?;
    check_child_fingerprint(&file, job)?;
    let ids = read_ids(&dir.join("token_ids.txt"))?;
    if file.sample.n_gen != ids.len() as u64 {
        return Err(format!(
            "{}: metrics n_gen {} != token_ids {}",
            dir.display(),
            file.sample.n_gen,
            ids.len()
        ));
    }
    if file.sample.m_peak_kib.is_none() {
        return Err(format!(
            "{}: M_peak is missing (VmRSS). Refusing a cost card without it.",
            dir.display()
        ));
    }
    Ok(ChildRun {
        sample: file.sample,
        ids,
        gemv_pack: file.header.gemv_pack,
    })
}

fn check_child_fingerprint(file: &SampleFile, job: &ArenaJob) -> Result<(), String> {
    if file.header.threads != job.threads as u64 {
        return Err(format!(
            "child threads {} != requested {}",
            file.header.threads, job.threads
        ));
    }
    if file.header.batch != 1 {
        return Err(format!(
            "child batch is {}, showdown batch is 1",
            file.header.batch
        ));
    }
    if file.header.quant_or_dtype != "bf16" {
        return Err(format!(
            "child quant_or_dtype is {}",
            file.header.quant_or_dtype
        ));
    }
    if file.header.acc_path != "bf16_w_fp32_acc" {
        return Err(format!("child acc_path is {}", file.header.acc_path));
    }
    if file.sample.n_gen != job.steps as u64 {
        return Err(format!(
            "child n_gen {} != --steps {}",
            file.sample.n_gen, job.steps
        ));
    }
    Ok(())
}

fn summarize(
    job: &ArenaJob,
    gemv_pack: bool,
    reference: &[u32],
    warmup_samples: &[PhaseSample],
    warmup_ids: &[Vec<u32>],
    scored_samples: &[PhaseSample],
    scored_ids: &[Vec<u32>],
) -> Result<ArenaReport, String> {
    let det_pass = ids_identical(scored_ids)
        && scored_ids.len() == job.repeats
        && scored_ids.iter().all(|ids| ids.len() == job.steps);
    let correct_pass = det_pass && scored_ids.iter().all(|ids| ids == reference);
    let warmup_match = ids_identical(warmup_ids)
        && (warmup_ids.is_empty() || scored_ids.first().is_some_and(|ids| warmup_ids[0] == *ids));
    let medians = meter::medians_of(scored_samples)
        .ok_or_else(|| "scored runs did not yield T_decode / L_* / M_peak medians".to_string())?;
    let comparable = job.threads as u64 == LLAMA_S1_BF16_THREADS && !gemv_pack;
    let gates_pass = correct_pass && det_pass;
    let verdict = meter::verdict(Some(&medians), comparable, gates_pass);
    let n_prompt = scored_samples[0].n_prompt;
    let fingerprint = job
        .out_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("s1-myos-bf16")
        .to_string();
    let card = render_cost_card(&CostCard {
        fingerprint,
        threads: job.threads as u64,
        batch: 1,
        gemv_pack,
        n_prompt,
        n_gen: job.steps as u64,
        n_ctx: n_prompt + job.steps as u64,
        warmup: job.warmup,
        repeats: job.repeats,
        commit: job.commit.clone(),
        cmdline: job.cmdline.clone(),
        correct: if correct_pass { "PASS" } else { "FAIL" },
        det: if det_pass { "PASS" } else { "FAIL" },
        medians: Some(medians.clone()),
        measured_on_this_machine: true,
        extra_note: format!(
            "warmup ids match scored: {warmup_match}. Per-run dirs: warm-* discarded, score-* scored. Child `metrics.json` is one process; this directory's `metrics.json` is the W/R aggregate."
        ),
    });
    write_text(&job.out_dir.join("cost-card.md"), &card)
        .map_err(|err| format!("cost-card: {err}"))?;
    write_text(&job.cost_card, &card)
        .map_err(|err| format!("{}: {err}", job.cost_card.display()))?;
    let ids = scored_ids.first().cloned().unwrap_or_default();
    write_token_ids(&job.out_dir, &ids).map_err(|err| format!("token_ids: {err}"))?;
    write_text(&job.out_dir.join("tokens.txt"), &join_ids(&ids))
        .map_err(|err| format!("tokens.txt: {err}"))?;
    write_text(
        &job.out_dir.join("metrics.json"),
        &aggregate_json(
            job,
            gemv_pack,
            scored_samples,
            warmup_samples,
            &medians,
            det_pass,
            correct_pass,
        ),
    )
    .map_err(|err| format!("metrics.json: {err}"))?;
    write_text(
        &job.out_dir.join("det_evidence.json"),
        &det_json(scored_ids, warmup_ids, det_pass, correct_pass),
    )
    .map_err(|err| format!("det_evidence.json: {err}"))?;
    Ok(ArenaReport {
        det_pass,
        correct_pass,
        comparable,
        scene: verdict.scene,
        medians,
    })
}

fn aggregate_json(
    job: &ArenaJob,
    gemv_pack: bool,
    scored: &[PhaseSample],
    warmup: &[PhaseSample],
    medians: &ShowdownMedians,
    det_pass: bool,
    correct_pass: bool,
) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str("  \"scenario_id\": \"S1\",\n");
    out.push_str("  \"impl\": \"myos\",\n");
    out.push_str("  \"quant_or_dtype\": \"bf16\",\n");
    out.push_str("  \"acc_path\": \"bf16_w_fp32_acc\",\n");
    out.push_str(&format!(
        "  \"gemv_pack\": {},\n",
        if gemv_pack { "true" } else { "false" }
    ));
    out.push_str(&format!("  \"threads\": {},\n", job.threads));
    out.push_str("  \"batch\": 1,\n");
    out.push_str(&format!("  \"W\": {},\n", job.warmup));
    out.push_str(&format!("  \"R\": {},\n", job.repeats));
    out.push_str(&format!(
        "  \"det\": \"{}\",\n",
        if det_pass { "PASS" } else { "FAIL" }
    ));
    out.push_str(&format!(
        "  \"CORRECT\": \"{}\",\n",
        if correct_pass { "PASS" } else { "FAIL" }
    ));
    out.push_str("  \"median\": {\n");
    out.push_str(&format!(
        "    \"T_decode_tok_s\": {:.6},\n",
        medians.t_decode_tok_s
    ));
    out.push_str(&format!("    \"L_TTFT_ms\": {:.6},\n", medians.l_ttft_ms));
    out.push_str(&format!("    \"L_TPOT_ms\": {:.6},\n", medians.l_tpot_ms));
    out.push_str(&format!("    \"M_peak_MiB\": {:.6},\n", medians.m_peak_mib));
    out.push_str(&format!(
        "    \"T_e2e_tok_s\": {:.6},\n",
        medians.t_e2e_tok_s
    ));
    out.push_str(&format!("    \"t_load_s\": {:.6}\n", medians.t_load_s));
    out.push_str("  },\n");
    out.push_str("  \"min\": ");
    push_span(&mut out, scored, false);
    out.push_str(",\n  \"max\": ");
    push_span(&mut out, scored, true);
    out.push_str(",\n  \"per_run_scored\": ");
    push_runs(&mut out, scored);
    out.push_str(",\n  \"per_run_warmup_discarded\": ");
    push_runs(&mut out, warmup);
    out.push_str(",\n");
    out.push_str(&format!(
        "  \"llama_baseline_run\": \"{}\"\n",
        meter::LLAMA_S1_BF16_RUN
    ));
    out.push_str("}\n");
    out
}

fn push_span(out: &mut String, samples: &[PhaseSample], max: bool) {
    let derived: Vec<_> = samples.iter().map(meter::derive).collect();
    let pick = |values: Vec<f64>| -> Option<f64> {
        values
            .into_iter()
            .reduce(|a, b| if max { a.max(b) } else { a.min(b) })
    };
    out.push_str("{\n");
    out.push_str(&format!(
        "    \"T_decode_tok_s\": {},\n",
        fmt_opt(pick(
            derived.iter().filter_map(|d| d.t_decode_tok_s).collect()
        ))
    ));
    out.push_str(&format!(
        "    \"L_TTFT_ms\": {},\n",
        fmt_opt(pick(derived.iter().map(|d| d.l_ttft_ms).collect()))
    ));
    out.push_str(&format!(
        "    \"L_TPOT_ms\": {},\n",
        fmt_opt(pick(derived.iter().filter_map(|d| d.l_tpot_ms).collect()))
    ));
    out.push_str(&format!(
        "    \"M_peak_MiB\": {}\n",
        fmt_opt(pick(derived.iter().filter_map(|d| d.m_peak_mib).collect()))
    ));
    out.push_str("  }");
}

fn push_runs(out: &mut String, samples: &[PhaseSample]) {
    out.push_str("[\n");
    for (i, sample) in samples.iter().enumerate() {
        let d = meter::derive(sample);
        out.push_str("    {");
        out.push_str(&format!("\"t_load_s\": {:.6}, ", sample.t_load_s));
        out.push_str(&format!("\"t_prefill_s\": {:.6}, ", sample.t_prefill_s));
        out.push_str(&format!("\"t_decode_s\": {:.6}, ", sample.t_decode_s));
        out.push_str(&format!(
            "\"T_decode_tok_s\": {}, ",
            fmt_opt(d.t_decode_tok_s)
        ));
        out.push_str(&format!("\"L_TTFT_ms\": {:.6}, ", d.l_ttft_ms));
        out.push_str(&format!("\"L_TPOT_ms\": {}, ", fmt_opt(d.l_tpot_ms)));
        out.push_str(&format!("\"M_peak_MiB\": {}, ", fmt_opt(d.m_peak_mib)));
        out.push_str(&format!("\"n_prompt\": {}, ", sample.n_prompt));
        out.push_str(&format!("\"n_gen\": {}", sample.n_gen));
        out.push('}');
        if i + 1 != samples.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("  ]");
}

fn fmt_opt(value: Option<f64>) -> String {
    match value {
        Some(v) => format!("{v:.6}"),
        None => "null".to_string(),
    }
}

fn det_json(
    scored: &[Vec<u32>],
    warmup: &[Vec<u32>],
    det_pass: bool,
    correct_pass: bool,
) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!(
        "  \"det\": \"{}\",\n",
        if det_pass { "PASS" } else { "FAIL" }
    ));
    out.push_str(&format!(
        "  \"CORRECT\": \"{}\",\n",
        if correct_pass { "PASS" } else { "FAIL" }
    ));
    out.push_str(&format!(
        "  \"ids_identical\": {},\n",
        if ids_identical(scored) {
            "true"
        } else {
            "false"
        }
    ));
    out.push_str("  \"lens\": [");
    for (i, ids) in scored.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&ids.len().to_string());
    }
    out.push_str("],\n");
    out.push_str("  \"first_diff\": ");
    match meter::first_id_diff(scored) {
        None => out.push_str("null"),
        Some((group, index, a, b)) => out.push_str(&format!(
            "{{\"vs_score\": {group}, \"index\": {index}, \"score0\": {}, \"other\": {}}}",
            fmt_id(a),
            fmt_id(b)
        )),
    }
    out.push_str(",\n  \"warmup_lens\": [");
    for (i, ids) in warmup.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&ids.len().to_string());
    }
    out.push_str("]\n}\n");
    out
}

fn fmt_id(id: Option<u32>) -> String {
    match id {
        Some(id) => id.to_string(),
        None => "null".to_string(),
    }
}

fn join_ids(ids: &[u32]) -> String {
    let mut text = String::new();
    for (i, id) in ids.iter().enumerate() {
        if i > 0 {
            text.push(' ');
        }
        text.push_str(&id.to_string());
    }
    text.push('\n');
    text
}

fn read_ids(path: &Path) -> Result<Vec<u32>, String> {
    let text = fs::read_to_string(path).map_err(|err| format!("{}: {err}", path.display()))?;
    let mut ids = Vec::new();
    for token in text.split_whitespace() {
        let id = token
            .parse::<u32>()
            .map_err(|_| format!("{}: `{token}` is not a u32", path.display()))?;
        ids.push(id);
    }
    if ids.is_empty() {
        return Err(format!("{}: file is empty", path.display()));
    }
    Ok(ids)
}

fn stderr_tail(stderr: &str) -> String {
    let lines: Vec<&str> = stderr.lines().collect();
    let start = lines.len().saturating_sub(12);
    lines[start..].join(" | ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meter::PhaseSample;

    fn sample() -> PhaseSample {
        PhaseSample {
            t_load_s: 1.0,
            t_prefill_s: 0.5,
            t_decode_s: 8.0,
            n_prompt: 128,
            n_gen: 2,
            m_peak_kib: Some(1024),
            m_rss_after_load_kib: Some(512),
            correct: Some(true),
        }
    }

    #[test]
    fn summarize_writes_det_pass_and_a_loss_against_llama() {
        let dir = std::env::temp_dir().join(format!(
            "myos-arena-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let card = dir.join("card.md");
        let job = ArenaJob {
            bin: PathBuf::from("myos"),
            weights: "w".into(),
            prompt_ids: "p".into(),
            ref_ids: "r".into(),
            steps: 2,
            threads: 8,
            warmup: 1,
            repeats: 3,
            out_dir: dir.join("run"),
            cost_card: card.clone(),
            cmdline: "myos --arena-s1 --threads 8".into(),
            commit: Some("abc".into()),
        };
        let ids = vec![vec![7, 8], vec![7, 8], vec![7, 8]];
        let samples = vec![sample(), sample(), sample()];
        let report = summarize(
            &job,
            false,
            &[7, 8],
            &[sample()],
            &[vec![7, 8]],
            &samples,
            &ids,
        )
        .unwrap();
        assert!(report.det_pass);
        assert!(report.correct_pass);
        assert!(report.comparable);
        assert_eq!(report.scene, SceneLabel::Scored(crate::meter::Band::Loss));
        let metrics = fs::read_to_string(job.out_dir.join("metrics.json")).unwrap();
        assert!(metrics.contains("\"det\": \"PASS\""), "{metrics}");
        assert!(metrics.contains("\"CORRECT\": \"PASS\""), "{metrics}");
        assert!(metrics.contains("\"W\": 1"), "{metrics}");
        assert!(metrics.contains("\"R\": 3"), "{metrics}");
        assert!(metrics.contains("\"gemv_pack\": false"), "{metrics}");
        assert!(
            metrics.contains("\"acc_path\": \"bf16_w_fp32_acc\""),
            "{metrics}"
        );
        assert_eq!(
            fs::read_to_string(job.out_dir.join("token_ids.txt")).unwrap(),
            "7 8\n"
        );
        assert_eq!(
            fs::read_to_string(job.out_dir.join("tokens.txt")).unwrap(),
            "7 8\n"
        );
        let card_text = fs::read_to_string(&card).unwrap();
        assert!(card_text.contains("场景负"), "{card_text}");
        assert!(card_text.contains("17.2348"), "{card_text}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn det_fail_when_scored_ids_differ() {
        let dir = std::env::temp_dir().join(format!(
            "myos-arena-det-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let job = ArenaJob {
            bin: PathBuf::from("myos"),
            weights: "w".into(),
            prompt_ids: "p".into(),
            ref_ids: "r".into(),
            steps: 2,
            threads: 8,
            warmup: 0,
            repeats: 2,
            out_dir: dir.join("run"),
            cost_card: dir.join("card.md"),
            cmdline: "myos".into(),
            commit: None,
        };
        let report = summarize(
            &job,
            false,
            &[1, 2],
            &[],
            &[],
            &[sample(), sample()],
            &[vec![1, 2], vec![1, 9]],
        )
        .unwrap();
        assert!(!report.det_pass);
        assert!(!report.correct_pass);
        assert_eq!(report.scene, SceneLabel::GateLoss);
        let evidence = fs::read_to_string(job.out_dir.join("det_evidence.json")).unwrap();
        assert!(evidence.contains("\"det\": \"FAIL\""), "{evidence}");
        assert!(evidence.contains("\"index\": 1"), "{evidence}");
        let _ = fs::remove_dir_all(&dir);
    }
}
