#![recursion_limit = "256"]

use proofpatch_core as plc;
use schemars::JsonSchema;
use serde_json::json;
use similar::TextDiff;
use std::io::Read as _;
use std::path::PathBuf;
use std::time::Duration as StdDuration;
use std::{fs, io};
use tempfile::NamedTempFile;

fn extract_json_from_text(s: &str) -> Option<serde_json::Value> {
    plc::json_extract::extract_first_json_value(s)
}

#[cfg(feature = "axi-agent")]
fn default_context_lines() -> u64 {
    8
}
#[cfg(feature = "axi-agent")]
fn default_nearby_lines() -> u64 {
    120
}
#[cfg(feature = "axi-agent")]
fn default_max_nearby() -> u64 {
    30
}
#[cfg(feature = "axi-agent")]
fn default_max_imports() -> u64 {
    30
}

fn truncate_str(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max_chars).collect();
    out.push_str("…");
    out
}

fn cap_string(s: &str, max_chars: usize) -> String {
    let s = s.trim();
    if s.is_empty() {
        return String::new();
    }
    truncate_str(s, max_chars)
}

fn cap_string_list(xs: &mut Vec<String>, max_items: usize, max_chars: usize) {
    // Trim, drop empties, dedupe (preserve first occurrence), then cap.
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for x in xs.drain(..) {
        let x = cap_string(&x, max_chars);
        if x.is_empty() {
            continue;
        }
        if seen.insert(x.clone()) {
            out.push(x);
        }
        if out.len() >= max_items {
            break;
        }
    }
    *xs = out;
}

fn cap_urls(urls: &mut Vec<String>, max_items: usize, max_chars: usize) {
    cap_string_list(urls, max_items, max_chars);
    urls.retain(|u| u.starts_with("http"));
}

fn cap_research_top(top: &mut Vec<ResearchTop>, max_top: usize, max_chars: usize) {
    if top.len() > max_top {
        top.truncate(max_top);
    }
    for t in top.iter_mut() {
        t.title = cap_string(&t.title, max_chars);
        t.why = cap_string(&t.why, max_chars * 2);
        cap_urls(&mut t.urls, 3, max_chars);
    }
}

fn cap_summary_v1(
    mut s: ResearchSummary,
    preset: &plc::config::ResearchPresetResolved,
) -> ResearchSummary {
    cap_research_top(&mut s.top, preset.llm_max_top, preset.llm_max_str_chars);
    cap_string_list(
        &mut s.math_keywords,
        preset.llm_max_list_items,
        preset.llm_max_str_chars,
    );
    cap_string_list(
        &mut s.mathlib_search,
        preset.llm_max_list_items,
        preset.llm_max_str_chars,
    );
    cap_string_list(
        &mut s.proof_shape,
        preset.llm_max_list_items,
        preset.llm_max_str_chars * 2,
    );
    cap_string_list(
        &mut s.pitfalls,
        preset.llm_max_list_items,
        preset.llm_max_str_chars * 2,
    );
    s
}

fn cap_summary_v2(
    mut s: ResearchSummaryV2,
    preset: &plc::config::ResearchPresetResolved,
) -> ResearchSummaryV2 {
    cap_research_top(&mut s.top, preset.llm_max_top, preset.llm_max_str_chars);
    cap_string_list(
        &mut s.math_keywords,
        preset.llm_max_list_items,
        preset.llm_max_str_chars,
    );
    cap_string_list(
        &mut s.mathlib_idents,
        preset.llm_max_list_items,
        preset.llm_max_str_chars,
    );
    // Very light sanity filter: Lean-ish identifiers only.
    s.mathlib_idents.retain(|id| {
        !id.is_empty()
            && id.len() <= preset.llm_max_str_chars
            && id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '\''))
    });
    cap_string_list(
        &mut s.search_queries,
        preset.llm_max_list_items,
        preset.llm_max_str_chars,
    );
    cap_string_list(
        &mut s.proof_shape,
        preset.llm_max_list_items,
        preset.llm_max_str_chars * 2,
    );
    cap_string_list(
        &mut s.pitfalls,
        preset.llm_max_list_items,
        preset.llm_max_str_chars * 2,
    );
    s
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(deny_unknown_fields)]
struct ResearchSummary {
    pub top: Vec<ResearchTop>,
    #[schemars(required)]
    #[serde(default)]
    pub math_keywords: Vec<String>,
    #[schemars(required)]
    #[serde(default)]
    pub mathlib_search: Vec<String>,
    #[schemars(required)]
    #[serde(default)]
    pub proof_shape: Vec<String>,
    #[schemars(required)]
    #[serde(default)]
    pub pitfalls: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(deny_unknown_fields)]
struct ResearchTop {
    pub title: String,
    pub why: String,
    #[schemars(required)]
    #[serde(default)]
    pub urls: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(deny_unknown_fields)]
struct ResearchSummaryV2 {
    pub top: Vec<ResearchTop>,
    /// Mathematical keywords (paper/theory side).
    #[schemars(required)]
    #[serde(default)]
    pub math_keywords: Vec<String>,
    /// Mathlib identifiers (Lean side), suitable for `#check`/`#find?` seeds.
    #[schemars(required)]
    #[serde(default)]
    pub mathlib_idents: Vec<String>,
    /// Search strings (not code) to try in Lean or grep (e.g. "GramSchmidt norm_sq").
    #[schemars(required)]
    #[serde(default)]
    pub search_queries: Vec<String>,
    #[schemars(required)]
    #[serde(default)]
    pub proof_shape: Vec<String>,
    #[schemars(required)]
    #[serde(default)]
    pub pitfalls: Vec<String>,
}

fn research_summary_kind_default() -> &'static str {
    "formalization_v1"
}

fn normalize_summary_kind(s: &str) -> String {
    s.trim().to_lowercase()
}

fn research_summary_system_prompt(kind: &str) -> String {
    let kind = normalize_summary_kind(kind);
    match kind.as_str() {
        "formalization_v2" => [
            "You are a research assistant for Lean/mathlib formalization.",
            "You will be given a small list of arXiv papers (title/abstract).",
            "CRITICAL:",
            "- Return STRICT JSON only (no markdown, no fences).",
            "- Do NOT invent Lean theorem statements or pseudo-code like `theorem foo : ...`.",
            "- Prefer mathlib identifiers that actually exist (or are plausible) and search queries we can run.",
        ]
        .join("\n"),
        _ => [
            "You are a research assistant for Lean/mathlib formalization.",
            "Given a small list of arXiv papers (title/abstract), extract what helps formalization.",
            "CRITICAL: do NOT invent Lean theorem statements or pseudo-code like `theorem foo : ...`.",
            "Instead, output mathlib-searchable keywords and concrete proof-shape notes.",
            "Return STRICT JSON (no markdown) with keys exactly:",
            r#"{"top":[{"title":"...","why":"...","urls":["..."]}],"math_keywords":["..."],"mathlib_search":["..."],"proof_shape":["..."],"pitfalls":["..."]}"#,
        ]
        .join("\n"),
    }
}

// NOTE: Advanced “agent tool-calling” mode (axi-based) is not part of the public, standalone
// build of proofpatch-core. Keep `proofpatch` usable without any extra workspace crates.

#[cfg(feature = "axi-agent")]
#[derive(Debug, Clone, serde::Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(deny_unknown_fields)]
struct ContextPackArgs {
    file: String,
    // OpenAI-strict schemas want every property listed in `required`. We keep these as Option
    // but mark them `required` so the schema includes them, and allow `null` when unset.
    #[schemars(required)]
    #[serde(default)]
    decl: Option<String>,
    #[schemars(required)]
    #[serde(default)]
    line: Option<u64>,
    #[schemars(required)]
    #[serde(default = "default_context_lines")]
    context_lines: u64,
    #[schemars(required)]
    #[serde(default = "default_nearby_lines")]
    nearby_lines: u64,
    #[schemars(required)]
    #[serde(default = "default_max_nearby")]
    max_nearby: u64,
    #[schemars(required)]
    #[serde(default = "default_max_imports")]
    max_imports: u64,
}

#[cfg(feature = "axi-agent")]
fn default_timeout_s() -> u64 {
    90
}

#[cfg(feature = "axi-agent")]
#[derive(Debug, Clone, serde::Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(deny_unknown_fields)]
struct VerifySummaryArgs {
    file: String,
    #[schemars(required)]
    #[serde(default = "default_timeout_s")]
    timeout_s: u64,
}

#[cfg(feature = "axi-agent")]
fn default_sorry_context_lines() -> u64 {
    1
}

#[cfg(feature = "axi-agent")]
fn default_max_sorries() -> u64 {
    30
}

#[cfg(feature = "axi-agent")]
#[derive(Debug, Clone, serde::Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(deny_unknown_fields)]
struct LocateSorriesArgs {
    file: String,
    #[schemars(required)]
    #[serde(default = "default_max_sorries")]
    max_sorries: u64,
    #[schemars(required)]
    #[serde(default = "default_sorry_context_lines")]
    context_lines: u64,
}

#[cfg(feature = "axi-agent")]
fn default_patch_timeout_s() -> u64 {
    120
}

#[cfg(feature = "axi-agent")]
fn default_patch_verify() -> bool {
    true
}

#[cfg(feature = "axi-agent")]
fn default_patch_write() -> bool {
    false
}

#[cfg(feature = "axi-agent")]
#[derive(Debug, Clone, serde::Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(deny_unknown_fields)]
struct PatchDeclArgs {
    file: String,
    decl: String,
    replacement: String,
    /// If true, write the patched text back to the file on disk.
    /// If false, patch and verify in-memory only.
    #[schemars(required)]
    #[serde(default = "default_patch_write")]
    write: bool,
    /// If true, run Lean verification after patching.
    #[schemars(required)]
    #[serde(default = "default_patch_verify")]
    verify: bool,
    #[schemars(required)]
    #[serde(default = "default_patch_timeout_s")]
    timeout_s: u64,
}

fn arg_value(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn arg_values(args: &[String], key: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < args.len() {
        if args[i] == key {
            if let Some(v) = args.get(i + 1) {
                out.push(v.clone());
            }
            i = i.saturating_add(2);
            continue;
        }
        i += 1;
    }
    out
}

fn arg_flag(args: &[String], key: &str) -> bool {
    args.iter().any(|a| a == key)
}

fn load_hint_rules(
    repo_root: &std::path::Path,
    research_preset: Option<&str>,
) -> Vec<plc::config::HintRule> {
    let Ok(Some(cfg)) = plc::config::load_from_repo_root(repo_root) else {
        return Vec::new();
    };

    // Start from repo defaults (proofpatch remains repo-agnostic; the repo opts in).
    let mut enabled: Vec<String> = cfg.hints.defaults.enabled_packs.clone();

    // If a research preset provides explicit packs, treat it as an override.
    if let Some(preset_name) = research_preset {
        if let Some(preset) = cfg.research.resolve_preset(preset_name) {
            if let Some(ts) = preset.tree_search {
                if let Some(packs) = ts.hint_packs {
                    enabled = packs;
                }
            }
        }
    }

    let mut rules: Vec<plc::config::HintRule> = Vec::new();
    for name in enabled {
        if let Some(pack) = cfg.hints.packs.get(&name) {
            rules.extend(pack.rules.clone());
        }
    }
    rules
}

fn env_truthy(name: &str, default_on: bool) -> bool {
    let v = std::env::var(name).ok().unwrap_or_default();
    let v = v.trim().to_lowercase();
    if v.is_empty() {
        return default_on;
    }
    !matches!(v.as_str(), "0" | "false" | "no" | "off")
}

fn arg_u64(args: &[String], key: &str) -> Option<u64> {
    arg_value(args, key).and_then(|s| s.trim().parse::<u64>().ok())
}

fn write_json(path: &std::path::Path, value: &serde_json::Value) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create dir {}: {}", parent.display(), e))?;
    }
    let mut f = fs::File::create(path)
        .map_err(|e| format!("failed to create {}: {}", path.display(), e))?;
    let s = serde_json::to_string_pretty(value).map_err(|e| format!("json encode: {e}"))?;
    io::Write::write_all(&mut f, s.as_bytes())
        .map_err(|e| format!("failed to write {}: {}", path.display(), e))?;
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

fn read_json(path: &std::path::Path) -> Option<serde_json::Value> {
    let s = std::fs::read_to_string(path).ok()?;
    serde_json::from_str::<serde_json::Value>(&s).ok()
}

fn durable_atomic_write(cache_root: &std::path::Path, rel: &str, data: &[u8]) {
    // Standalone atomic-ish write: write temp file then rename.
    // We intentionally avoid extra workspace dependencies here.
    let p = cache_root.join(rel);
    if let Some(parent) = p.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let Some(parent) = p.parent() else {
        let _ = std::fs::write(p, data);
        return;
    };
    match NamedTempFile::new_in(parent) {
        Ok(mut tmp) => {
            let _ = std::io::Write::write_all(&mut tmp, data);
            let _ = tmp.persist(&p);
        }
        Err(_) => {
            let _ = std::fs::write(p, data);
        }
    }
}

fn hyp_name_from_text_line(s: &str) -> Option<String> {
    let (lhs, _) = s.split_once(':')?;
    let nm = lhs.trim().split_whitespace().next()?.trim();
    if nm.is_empty() {
        return None;
    }
    if nm
        .chars()
        .next()
        .map(|c| c.is_ascii_alphabetic() || c == '_')
        .unwrap_or(false)
        && nm.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        Some(nm.to_string())
    } else {
        None
    }
}

fn smt_support_lean_hints(
    core: &serde_json::Value,
    ctx_hyp_names: &std::collections::HashSet<String>,
) -> Option<Vec<String>> {
    let items = core.get("core_items")?.as_array()?;
    let mut srcs: Vec<String> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    for it in items {
        let name = it.get("name").and_then(|v| v.as_str()).unwrap_or("").trim();
        if name == "neg_target" {
            continue;
        }
        let is_ident = name
            .chars()
            .next()
            .map(|c| c.is_ascii_alphabetic() || c == '_')
            .unwrap_or(false)
            && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        // Only include hypothesis names that actually exist in the Lean local context.
        if is_ident && ctx_hyp_names.contains(name) {
            names.push(name.to_string());
        }
        let src = it.get("src")?.as_str()?.trim();
        if !src.is_empty() {
            srcs.push(src.to_string());
        }
    }
    srcs.sort();
    srcs.dedup();
    srcs.truncate(12);
    names.sort();
    names.dedup();

    let mut comment = String::new();
    if !srcs.is_empty() {
        comment.push_str("  -- SMT support set:\n");
        for s in &srcs {
            comment.push_str(&format!("  --   * {s}\n"));
        }
    } else {
        comment.push_str("  -- SMT support set: (unavailable)\n");
    }
    let bracketed = if names.is_empty() {
        String::new()
    } else {
        format!(" [{}]", names.join(", "))
    };

    // “Support-guided proof search”: emit a small, bounded portfolio of tactic scripts.
    // The caller will verify and pick what works.
    let mut out: Vec<String> = Vec::new();
    out.push(format!(
        "by\n  -- proofpatch:smt_support\n{comment}  try (norm_cast; done)\n  try (linarith{bracketed}; done)\n  try (nlinarith{bracketed}; done)\n  try (omega; done)"
    ));
    out.push(format!(
        "by\n  -- proofpatch:smt_support\n{comment}  try (simp; norm_cast; done)\n  try (simp; linarith{bracketed}; done)\n  try (simp; nlinarith{bracketed}; done)\n  try (simp; omega; done)"
    ));
    out.push(format!(
        "by\n  -- proofpatch:smt_support\n{comment}  try (omega; done)\n  try (linarith{bracketed}; done)\n  try (nlinarith{bracketed}; done)\n  try (norm_cast; done)"
    ));
    out.push(format!(
        "by\n  -- proofpatch:smt_support\n{comment}  try (nlinarith{bracketed}; done)\n  try (linarith{bracketed}; done)\n  try (omega; done)"
    ));
    out.push(format!(
        "by\n  -- proofpatch:smt_support\n{comment}  try (linarith{bracketed}; done)\n  try (omega; done)\n  try (nlinarith{bracketed}; done)"
    ));
    Some(out)
}

fn smt_dump_root(
    repo_root: &std::path::Path,
    smt_dump_dir_opt: &Option<std::path::PathBuf>,
) -> std::path::PathBuf {
    if let Some(p) = smt_dump_dir_opt.as_ref() {
        if p.is_absolute() {
            p.clone()
        } else {
            repo_root.join(p)
        }
    } else {
        repo_root.join(".generated").join("proofpatch-smt2")
    }
}

fn smt_proof_dump_root(
    repo_root: &std::path::Path,
    smt_proof_dump_dir_opt: &Option<std::path::PathBuf>,
) -> std::path::PathBuf {
    if let Some(p) = smt_proof_dump_dir_opt.as_ref() {
        if p.is_absolute() {
            p.clone()
        } else {
            repo_root.join(p)
        }
    } else {
        repo_root.join(".generated").join("proofpatch-smtproof")
    }
}

fn maybe_write_smt2_dump(
    root: &std::path::Path,
    state_key: u64,
    goal_sig: u64,
    depth: usize,
    script: &str,
) -> String {
    let rel = format!("smt2/{state_key}_{goal_sig}_d{depth}.smt2");
    durable_atomic_write(root, &rel, script.as_bytes());
    root.join(rel).display().to_string()
}

fn maybe_write_smt_proof_dump(
    root: &std::path::Path,
    state_key: u64,
    goal_sig: u64,
    depth: usize,
    proof_text: &str,
) -> String {
    let rel = format!("proof/{state_key}_{goal_sig}_d{depth}.sexp");
    durable_atomic_write(root, &rel, proof_text.as_bytes());
    root.join(rel).display().to_string()
}

fn cache_read_eval(
    cache_dir: &std::path::Path,
    key: u64,
    len: usize,
) -> Option<(serde_json::Value, serde_json::Value, usize, usize)> {
    // Include `len` in the filename to avoid hash-collision false hits.
    let dir = cache_dir.join("eval");
    let p_new = dir.join(format!("{key}_{len}.json"));
    let p_old = dir.join(format!("{key}.json"));
    let v = read_json(&p_new).or_else(|| read_json(&p_old))?;
    let got_len = v.get("len").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
    if got_len != len {
        return None;
    }
    let verify_raw = v.get("verify_raw")?.clone();
    let verify_summary = v.get("verify_summary")?.clone();
    let sorries = v.get("sorries").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
    let conservative = v
        .get("conservative_sorries")
        .and_then(|x| x.as_u64())
        .unwrap_or(0) as usize;
    Some((verify_raw, verify_summary, sorries, conservative))
}

fn cache_write_eval(
    cache_dir: &std::path::Path,
    key: u64,
    len: usize,
    verify_raw: &serde_json::Value,
    verify_summary: &serde_json::Value,
    sorries: usize,
    conservative_sorries: usize,
) {
    let rel = format!("eval/{key}_{len}.json");
    let v = json!({
        "len": len,
        "verify_raw": verify_raw,
        "verify_summary": verify_summary,
        "sorries": sorries,
        "conservative_sorries": conservative_sorries,
    });
    durable_atomic_write(cache_dir, &rel, v.to_string().as_bytes());
}

fn cache_read_goal_dump(
    cache_dir: &std::path::Path,
    text_hash: u64,
    len: usize,
    line: usize,
) -> Option<(u64, usize, usize, String)> {
    // Disk format (best-effort, forward-compatible):
    // - Stored under `goaldump/{text_hash}_{len}_{line}.json`
    // - May contain extra keys as we evolve the cache schema.
    //
    // Important: this cache is an *optimization only*. It must never affect correctness.
    let p = cache_dir
        .join("goaldump")
        .join(format!("{text_hash}_{len}_{line}.json"));
    let v = read_json(&p)?;
    let sk = v.get("state_key")?.as_u64()?;
    let ng = v.get("n_goals")?.as_u64()? as usize;
    let ht = v.get("hyps_total")?.as_u64()? as usize;
    let target = v
        .get("target")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    Some((sk, ng, ht, target))
}

fn cache_read_goal_dump_hyps_texts(
    cache_dir: &std::path::Path,
    text_hash: u64,
    len: usize,
    line: usize,
) -> Option<Vec<String>> {
    // We store a bounded list of hypothesis pretty-prints (`hyps_texts`) so that:
    // - SMT precheck can run *without additional Lean calls* (LIA-only, best-effort), and
    // - we can compute lightweight “coupling / context heaviness” hints in ranking.
    //
    // We cap to 48 lines to keep:
    // - disk cache stable and small,
    // - SMT inputs bounded,
    // - and JSON output readable.
    let p = cache_dir
        .join("goaldump")
        .join(format!("{text_hash}_{len}_{line}.json"));
    let v = read_json(&p)?;
    let xs = v.get("hyps_texts")?.as_array()?;
    let mut out = Vec::new();
    for x in xs {
        if let Some(s) = x.as_str() {
            let t = s.trim();
            if !t.is_empty() {
                out.push(t.to_string());
            }
        }
        if out.len() >= 48 {
            break;
        }
    }
    Some(out)
}

fn cache_write_goal_dump(
    cache_dir: &std::path::Path,
    text_hash: u64,
    len: usize,
    line: usize,
    state_key: u64,
    n_goals: usize,
    hyps_total: usize,
    target: &str,
    hyps_texts: &[String],
) {
    // Note: `target` is truncated for output/readability (not proof logic), while `hyps_texts`
    // is kept as a bounded list. This makes the cache suitable for:
    // - goal-first selection (cheap difficulty estimate),
    // - SMT precheck (LIA-only, best-effort),
    // - and explainable ranking hints (`rank_hint`).
    let rel = format!("goaldump/{text_hash}_{len}_{line}.json");
    let hyps_texts: Vec<String> = hyps_texts
        .iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .take(48)
        .collect();
    let v = json!({
        "text_hash": text_hash,
        "len": len,
        "line": line,
        "state_key": state_key,
        "n_goals": n_goals,
        "hyps_total": hyps_total,
        "target": truncate_str(target, 600),
        "hyps_texts": hyps_texts,
    });
    durable_atomic_write(cache_dir, &rel, v.to_string().as_bytes());
}

#[cfg(feature = "planner")]
fn cache_read_planner(
    cache_dir: &std::path::Path,
    state_key: u64,
    goal_sig: u64,
) -> Option<serde_json::Value> {
    let p = cache_dir
        .join("planner")
        .join(format!("{state_key}_{goal_sig}.json"));
    read_json(&p)
}

#[cfg(feature = "planner")]
fn cache_write_planner(
    cache_dir: &std::path::Path,
    state_key: u64,
    goal_sig: u64,
    v: &serde_json::Value,
) {
    let rel = format!("planner/{state_key}_{goal_sig}.json");
    durable_atomic_write(cache_dir, &rel, v.to_string().as_bytes());
}

fn cache_read_smt_entails(
    cache_dir: &std::path::Path,
    state_key: u64,
    goal_sig: u64,
    depth: usize,
) -> Option<bool> {
    let dir = cache_dir.join("smt");
    let p_new = dir.join(format!("{state_key}_{goal_sig}_d{depth}.json"));
    let v = read_json(&p_new).or_else(|| {
        // Back-compat: old cache key did not include depth. Only accept it for depth=0
        // since other depths have different semantics.
        if depth == 0 {
            let p_old = dir.join(format!("{state_key}_{goal_sig}.json"));
            read_json(&p_old)
        } else {
            None
        }
    })?;
    v.get("entails").and_then(|x| x.as_bool())
}

fn cache_write_smt_entails(
    cache_dir: &std::path::Path,
    state_key: u64,
    goal_sig: u64,
    depth: usize,
    entails: bool,
) {
    let rel = format!("smt/{state_key}_{goal_sig}_d{depth}.json");
    let v =
        json!({ "state_key": state_key, "goal_sig": goal_sig, "depth": depth, "entails": entails });
    durable_atomic_write(cache_dir, &rel, v.to_string().as_bytes());
}

// SMT/LIA parsing and entailment live in `proofpatch-core::smt_lia` (shared by CLI + MCP).

fn smt_entails_from_pp_dump_escalating(
    pp_dump: &serde_json::Value,
    timeout_ms: u64,
    seed: u64,
    depth: usize,
    solver_norm: &str,
    aggressive: bool,
    reuse: &mut Option<plc::smt_lia::ReusableSmtSession>,
    trace: &mut Vec<serde_json::Value>,
) -> Result<(Option<bool>, u64), String> {
    fn with_temp_env<T>(key: &str, val: Option<&str>, f: impl FnOnce() -> T) -> T {
        let old = std::env::var(key).ok();
        match val {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
        let out = f();
        match old.as_deref() {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
        out
    }

    fn fallback_solver_cmdlines(primary_norm: &str) -> Vec<&'static str> {
        // Keep this small + deterministic; we don't want to “spray solvers” by default.
        // Aggressive mode can afford a second solver try after a timeout ladder.
        let z3 = "z3 -in -smt2";
        let cvc5 = "cvc5 --lang smt2 --incremental";
        match primary_norm.trim().to_lowercase().as_str() {
            "cvc5" => vec![z3],
            "z3" => vec![cvc5],
            // auto/custom: prefer cvc5 as a distinct backend, then z3.
            _ => vec![cvc5, z3],
        }
    }

    let mut attempts: u64 = 0;
    let t1 = timeout_ms.max(1);
    attempts += 1;
    let res1 =
        plc::smt_lia::entails_from_pp_dump_with_depth_reuse(pp_dump, t1, seed, depth, reuse)?;
    if trace.len() < 64 {
        let reuse_solver = reuse
            .as_ref()
            .and_then(|s| s.stats().get("solver").cloned())
            .unwrap_or(serde_json::Value::Null);
        trace.push(json!({
            "step": "base",
            "timeout_ms": t1,
            "solver": reuse_solver,
            "outcome": if res1.is_some() { "decided" } else { "unknown" },
        }));
    }
    if res1.is_some() || !aggressive {
        return Ok((res1, attempts));
    }

    // Step 2: timeout ladder (same solver / reuse if possible).
    let t2 = (t1.saturating_mul(4)).min(30_000);
    attempts += 1;
    let res2 =
        plc::smt_lia::entails_from_pp_dump_with_depth_reuse(pp_dump, t2, seed, depth, reuse)?;
    if trace.len() < 64 {
        let reuse_solver = reuse
            .as_ref()
            .and_then(|s| s.stats().get("solver").cloned())
            .unwrap_or(serde_json::Value::Null);
        trace.push(json!({
            "step": "timeout_x4",
            "timeout_ms": t2,
            "solver": reuse_solver,
            "outcome": if res2.is_some() { "decided" } else { "unknown" },
        }));
    }
    if res2.is_some() {
        return Ok((res2, attempts));
    }

    // Step 3: solver fallback (per-call session; do not disrupt warm reuse).
    // This is the biggest “Yurichev-style” reliability bump: export/compare solvers.
    for cmd in fallback_solver_cmdlines(solver_norm) {
        attempts += 1;
        let res = with_temp_env("SMTKIT_SOLVER", Some(cmd), || {
            plc::smt_lia::entails_from_pp_dump_with_depth(pp_dump, t2, seed, depth)
        })?;
        if trace.len() < 64 {
            trace.push(json!({
                "step": "fallback_solver",
                "timeout_ms": t2,
                "solver": cmd,
                "outcome": if res.is_some() { "decided" } else { "unknown" },
            }));
        }
        if res.is_some() {
            return Ok((res, attempts));
        }
    }

    Ok((None, attempts))
}

fn smt_entails_from_hyps_target_escalating(
    hyps_texts: &[String],
    target: &str,
    timeout_ms: u64,
    seed: u64,
    depth: usize,
    solver_norm: &str,
    aggressive: bool,
    reuse: &mut Option<plc::smt_lia::ReusableSmtSession>,
    trace: &mut Vec<serde_json::Value>,
) -> Result<(Option<bool>, u64), String> {
    // Build a minimal pp_dump-shaped object so we can reuse the same logic.
    let mut pretty = String::new();
    for h in hyps_texts.iter().take(48) {
        pretty.push_str(h);
        pretty.push('\n');
    }
    pretty.push_str("⊢ ");
    pretty.push_str(target);

    let hyps_json: Vec<serde_json::Value> = hyps_texts
        .iter()
        .take(48)
        .map(|s| json!({ "text": s }))
        .collect();
    let pp_dump = json!({
        "goals": [{
            "pretty": pretty,
            "hyps": hyps_json
        }]
    });
    smt_entails_from_pp_dump_escalating(
        &pp_dump,
        timeout_ms,
        seed,
        depth,
        solver_norm,
        aggressive,
        reuse,
        trace,
    )
}

fn smt_explain_fragment_from_hyps_target(
    hyps_texts: &[String],
    target: &str,
    depth: usize,
    max_hyps: usize,
) -> Option<serde_json::Value> {
    if target.trim().is_empty() {
        return None;
    }
    let mut pretty = String::new();
    for h in hyps_texts.iter().take(48) {
        pretty.push_str(h);
        pretty.push('\n');
    }
    pretty.push_str("⊢ ");
    pretty.push_str(target);

    let hyps_json: Vec<serde_json::Value> = hyps_texts
        .iter()
        .take(48)
        .map(|s| json!({ "text": s }))
        .collect();
    let pp_dump = json!({
        "goals": [{
            "pretty": pretty,
            "hyps": hyps_json
        }]
    });
    plc::smt_lia::explain_fragment_from_pp_dump(&pp_dump, depth, max_hyps)
}

fn unified_diff_bounded(old: &str, new: &str, context: usize, max_chars: usize) -> (String, bool) {
    let diff = TextDiff::from_lines(old, new);
    let mut s = diff
        .unified_diff()
        .context_radius(context)
        .header("before", "after")
        .to_string();
    let truncated = s.chars().count() > max_chars;
    if truncated {
        s = s.chars().take(max_chars).collect::<String>();
        s.push_str("\n[proofpatch: diff truncated]\n");
    }
    (s, truncated)
}

fn verify_summary_from_raw_value(raw_v: &serde_json::Value) -> serde_json::Value {
    let ok = raw_v.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
    let timeout = raw_v
        .get("timeout")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let returncode = raw_v
        .get("returncode")
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    let stdout = raw_v.get("stdout").and_then(|v| v.as_str()).unwrap_or("");
    let stderr = raw_v.get("stderr").and_then(|v| v.as_str()).unwrap_or("");

    let first_error_loc =
        plc::parse_first_error_loc(stdout, stderr).and_then(|loc| serde_json::to_value(loc).ok());
    let errors = stdout.matches(": error:").count()
        + stdout.matches(": error(").count()
        + stderr.matches(": error:").count()
        + stderr.matches(": error(").count();
    let warnings = stdout.matches(": warning:").count()
        + stdout.matches(": warning(").count()
        + stderr.matches(": warning:").count()
        + stderr.matches(": warning(").count();
    // Lean linter: "warning: declaration uses 'sorry'" indicates the declaration is admitted even if
    // the source file contains no literal `sorry` token (e.g. via `apply?`/`simp?` suggestion tactics).
    let sorry_warnings = stdout.matches("declaration uses 'sorry'").count()
        + stderr.matches("declaration uses 'sorry'").count()
        + stdout.matches("declaration uses 'admit'").count()
        + stderr.matches("declaration uses 'admit'").count();

    json!({
        "ok": ok,
        "timeout": timeout,
        "returncode": returncode,
        "counts": { "errors": errors, "warnings": warnings, "sorry_warnings": sorry_warnings },
        "first_error": stdout
            .lines()
            .find(|l| l.contains(": error:") || l.contains(": error("))
            .or_else(|| stderr.lines().find(|l| l.contains(": error:") || l.contains(": error("))),
        "first_error_loc": first_error_loc
    })
}

fn usage() -> String {
    [
        "proofpatch — Lean proof debugging loop + SMT oracle.",
        "",
        "Core (Lean loop):",
        "  triage-file          --repo <path> --file <relpath> ...",
        "  verify-summary       --repo <path> --file <relpath> ...",
        "  locate-sorries       --repo <path> --file <relpath> ...",
        "  context-pack         --repo <path> --file <relpath> ...",
        "  patch|patch-region|patch-nearest   --repo <path> --file <relpath> ...",
        "  scratch-lemma        --repo <path> --file <relpath|module> --name <decl_name> ...",
        "",
        "SMT oracle (via smtkit):",
        "  smt-probe            [--output-json <path>]",
        "  smt-repro            --input-json <path|-> ...",
        "  tree-search-nearest  --repo <path> --file <relpath> ... (includes SMT knobs)",
        "",
        "Optional (LLM/research/review):",
        "  suggest | loop",
        "  arxiv-search | research-auto | research-ingest | research-attach",
        "  review-prompt | review-diff | llm-chat",
        "",
        "Other:",
        "  goal-dump-nearest | goal-analyze | goal-try",
        "  report | lint-style | agent-step | prompt | rubberduck-prompt",
        "  lean-embed-smoke (requires cargo feature `lean-embed`)",
        "",
        "Notes:",
        "- Output is JSON to stdout.",
        "- This CLI uses proofpatch-core, so verification runs `lake env lean` on the *real* file path.",
        "- HTML is optional; it’s intended for humans. Agents should consume the JSON table.",
    ]
    .join("\n")
}

fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(ch),
        }
    }
    out
}

fn apply_mechanical_fixes_for_first_error(
    text: &str,
    first_error_line_1: Option<usize>,
    first_error_text: Option<&str>,
) -> (String, Vec<serde_json::Value>) {
    let Some(line1) = first_error_line_1 else {
        return (text.to_string(), vec![]);
    };
    let Some(msg) = first_error_text else {
        return (text.to_string(), vec![]);
    };
    // Heuristic 1: if Lean suggests `ring_nf`, replace a nearby `ring` with `ring_nf`.
    if !msg.contains("ring_nf") {
        return (text.to_string(), vec![]);
    }

    let mut lines: Vec<String> = text.lines().map(|s| s.to_string()).collect();
    if lines.is_empty() {
        return (text.to_string(), vec![]);
    }

    let mut edits = Vec::new();
    let start0 = line1.saturating_sub(1).saturating_sub(40);
    let end0 = usize::min(lines.len().saturating_sub(1), (line1 - 1) + 60);
    for i0 in start0..=end0 {
        let ln = &lines[i0];
        let trimmed = ln.trim();
        if trimmed == "ring" {
            let before = ln.clone();
            let indent: String = ln.chars().take_while(|c| c.is_whitespace()).collect();
            lines[i0] = format!("{indent}ring_nf");
            edits.push(json!({
                "kind": "replace_tactic",
                "line": i0 + 1,
                "before": before,
                "after": lines[i0],
                "note": "Lean suggested ring_nf; replaced nearby `ring` with `ring_nf`.",
            }));
            break;
        }
    }

    (
        lines.join("\n") + if text.ends_with('\n') { "\n" } else { "" },
        edits,
    )
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("");
    let rest: &[String] = if args.len() > 2 { &args[2..] } else { &[] };

    if cmd.is_empty() || cmd == "--help" || cmd == "-h" || cmd == "help" {
        println!("{}", usage());
        return Ok(());
    }
    // Also accept `proofpatch <command> --help` (or `-h`/`help`) as a top-level usage request.
    // We don't currently render per-command usage; the global usage is still better than
    // "missing --repo" errors when users probe subcommands.
    if arg_flag(rest, "--help")
        || arg_flag(rest, "-h")
        || rest.first().map(|s| s.as_str()) == Some("help")
    {
        println!("{}", usage());
        return Ok(());
    }

    // Aliases / grouping:
    // - `proofpatch smt probe` == `proofpatch smt-probe`
    // - `proofpatch smt repro` == `proofpatch smt-repro`
    let (cmd, rest): (&str, &[String]) = if cmd == "smt" {
        let sub = rest.get(0).map(|s| s.as_str()).unwrap_or("");
        let tail: &[String] = if rest.len() > 1 { &rest[1..] } else { &[] };
        match sub {
            "probe" => ("smt-probe", tail),
            "repro" => ("smt-repro", tail),
            _ => {
                return Err(format!(
                    "{}\n\nsmt subcommands:\n  smt probe\n  smt repro\n",
                    usage()
                ));
            }
        }
    } else {
        (cmd, rest)
    };

    match cmd {
        "smt-probe" => {
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);
            let (sess, used, caps) = smtkit::session::spawn_auto_with_caps()
                .map_err(|e| format!("smt probe failed: {e}"))?;
            let _ = sess.exit();
            let out = json!({
                "ok": true,
                "solver": { "used": used },
                "caps": {
                    "check_sat_assuming": caps.check_sat_assuming,
                    "get_model": caps.get_model,
                    "get_unsat_core": caps.get_unsat_core,
                    "get_proof": caps.get_proof,
                    "named_assertions_in_core": caps.named_assertions_in_core,
                }
            });
            if let Some(p) = output_json {
                fs::write(&p, out.to_string())
                    .map_err(|e| format!("write {}: {e}", p.display()))?;
            }
            println!("{}", out);
            return Ok(());
        }

        "lean-embed-smoke" => {
            #[cfg(feature = "lean-embed")]
            {
                let r = proofpatch_lean_embed::add_u64(20, 22)?;
                println!(
                    "{}",
                    json!({
                        "ok": true,
                        "kind": "lean_embed_smoke",
                        "result": r,
                        "expected": 42
                    })
                    .to_string()
                );
                return Ok(());
            }
            #[cfg(not(feature = "lean-embed"))]
            {
                return Err("lean-embed-smoke requires building with: cargo run -p proofpatch-core --features lean-embed --bin proofpatch -- lean-embed-smoke".to_string());
            }
        }

        "triage-file" => {
            let repo_root = arg_value(rest, "--repo")
                .ok_or_else(|| "missing --repo".to_string())
                .map(PathBuf::from)?;
            let file = arg_value(rest, "--file").ok_or_else(|| "missing --file".to_string())?;
            let timeout_s = arg_u64(rest, "--timeout-s").unwrap_or(180);
            let max_sorries = arg_u64(rest, "--max-sorries").unwrap_or(50) as usize;
            let context_lines = arg_u64(rest, "--context-lines").unwrap_or(1) as usize;
            let include_context_pack = !arg_flag(rest, "--no-context-pack");
            let include_prompts = !arg_flag(rest, "--no-prompts");
            let include_raw_verify = arg_flag(rest, "--include-raw-verify");
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);
            let pack_context_lines = arg_u64(rest, "--pack-context-lines").unwrap_or(6) as usize;
            let pack_nearby_lines = arg_u64(rest, "--pack-nearby-lines").unwrap_or(60) as usize;
            let pack_max_nearby = arg_u64(rest, "--pack-max-nearby").unwrap_or(20) as usize;
            let pack_max_imports = arg_u64(rest, "--pack-max-imports").unwrap_or(20) as usize;

            let repo_root =
                plc::find_lean_repo_root(&repo_root).map_err(|e| format!("repo_root: {e}"))?;
            plc::load_dotenv_smart(&repo_root);

            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| format!("failed to build tokio runtime: {e}"))?;
            let raw = rt
                .block_on(plc::verify_lean_file(
                    &repo_root,
                    &file,
                    StdDuration::from_secs(timeout_s),
                ))
                .map_err(|e| format!("verify failed: {e}"))?;

            let raw_v = serde_json::to_value(raw).map_err(|e| format!("serialize verify: {e}"))?;
            let summary = verify_summary_from_raw_value(&raw_v);

            let locs = plc::locate_sorries_in_file(&repo_root, &file, max_sorries, context_lines)?;
            let conservative_sorries =
                plc::count_sorry_tokens_conservative_in_file(&repo_root, &file).unwrap_or(0);
            let selected_sorry = plc::select_primary_sorry(
                summary
                    .get("first_error_loc")
                    .and_then(|v| v.get("line"))
                    .and_then(|v| v.as_u64())
                    .map(|x| x as usize),
                &locs,
            );

            let (context_pack_first_error, context_pack_nearest_sorry) = if include_context_pack {
                let first_error_line = summary
                    .get("first_error_loc")
                    .and_then(|v| v.get("line"))
                    .and_then(|v| v.as_u64())
                    .map(|x| x as usize);

                let pack_first = first_error_line
                    .and_then(|line_1| {
                        plc::build_context_pack(
                            &repo_root,
                            &file,
                            None,
                            Some(line_1),
                            pack_context_lines,
                            pack_nearby_lines,
                            pack_max_nearby,
                            pack_max_imports,
                        )
                        .ok()
                    })
                    .and_then(|p| serde_json::to_value(p).ok());

                let pack_nearest = selected_sorry
                    .as_ref()
                    .and_then(|s| {
                        plc::build_context_pack(
                            &repo_root,
                            &file,
                            None,
                            Some(s.line),
                            pack_context_lines,
                            pack_nearby_lines,
                            pack_max_nearby,
                            pack_max_imports,
                        )
                        .ok()
                    })
                    .and_then(|p| serde_json::to_value(p).ok());

                (pack_first, pack_nearest)
            } else {
                (None, None)
            };

            let first_error_line = summary
                .get("first_error_loc")
                .and_then(|v| v.get("line"))
                .and_then(|v| v.as_u64())
                .map(|x| x as usize);
            let first_error_text = summary
                .get("first_error")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            let rubberduck_prompt_first_error = if include_prompts {
                first_error_line
                    .and_then(|line_1| {
                        plc::build_context_pack(
                            &repo_root,
                            &file,
                            None,
                            Some(line_1),
                            pack_context_lines,
                            pack_nearby_lines,
                            pack_max_nearby,
                            pack_max_imports,
                        )
                        .ok()
                    })
                    .and_then(|pack| {
                        let label = format!("line:{}", pack.focus.line.unwrap_or(0));
                        plc::build_rubberduck_prompt_from_excerpt(
                            &repo_root,
                            &file,
                            &label,
                            &pack.focus.excerpt,
                            first_error_text.as_deref(),
                        )
                        .ok()
                    })
                    .and_then(|p| serde_json::to_value(p).ok())
            } else {
                None
            };

            let patch_prompt_nearest_sorry = if include_prompts {
                selected_sorry
                    .as_ref()
                    .and_then(|s| {
                        plc::build_region_patch_prompt(
                            &repo_root,
                            &file,
                            s.region_start,
                            s.region_end,
                            first_error_text.as_deref(),
                        )
                        .ok()
                    })
                    .and_then(|p| serde_json::to_value(p).ok())
            } else {
                None
            };

            let next_action = {
                let has_error = summary.get("ok").and_then(|v| v.as_bool()) == Some(false)
                    && summary
                        .get("counts")
                        .and_then(|c| c.get("errors"))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0)
                        > 0;
                if has_error {
                    json!({
                        "kind": "fix_first_error",
                        "prompt": rubberduck_prompt_first_error,
                        "line": first_error_line,
                    })
                } else if selected_sorry.is_some() {
                    json!({
                        "kind": "patch_nearest_sorry",
                        "prompt": patch_prompt_nearest_sorry,
                        "region": selected_sorry.as_ref().map(|s| json!({"start_line": s.region_start, "end_line": s.region_end})),
                    })
                } else {
                    json!({ "kind": "noop" })
                }
            };

            let out = json!({
                "repo_root": repo_root.display().to_string(),
                "file": file,
                "verify": { "summary": summary, "raw": if include_raw_verify { raw_v } else { serde_json::Value::Null } },
                "sorries": { "count": locs.len(), "conservative_count": conservative_sorries, "locations": locs }
                ,
                "nearest_sorry_to_first_error": selected_sorry,
                "context_pack_first_error": context_pack_first_error,
                "context_pack_nearest_sorry": context_pack_nearest_sorry,
                "rubberduck_prompt_first_error": rubberduck_prompt_first_error,
                "patch_prompt_nearest_sorry": patch_prompt_nearest_sorry,
                "next_action": next_action
            });

            if let Some(p) = output_json {
                write_json(&p, &out)?;
                let small = json!({
                    "ok": true,
                    "written": p.display().to_string(),
                    "kind": "triage_file",
                    "result_kind": serde_json::Value::Null,
                    "repo_root": repo_root.display().to_string(),
                    "file": file,
                    "verify_ok": summary.get("ok").cloned().unwrap_or(serde_json::Value::Null),
                    "errors": summary.get("counts").and_then(|c| c.get("errors")).cloned().unwrap_or(serde_json::Value::Null),
                    "sorries": locs.len(),
                    "next_action": out.get("next_action").cloned().unwrap_or(serde_json::Value::Null),
                });
                println!("{}", small.to_string());
            } else {
                println!("{}", out.to_string());
            }
            Ok(())
        }

        "goal-dump-nearest" => {
            let repo_root = arg_value(rest, "--repo")
                .ok_or_else(|| "missing --repo".to_string())
                .map(PathBuf::from)?;
            let file = arg_value(rest, "--file").ok_or_else(|| "missing --file".to_string())?;
            let timeout_s = arg_u64(rest, "--timeout-s").unwrap_or(12);
            let focus_line = arg_u64(rest, "--focus-line").map(|x| x as usize);
            let focus_decl = arg_value(rest, "--focus-decl");
            let pp_dump_only = arg_flag(rest, "--pp-dump-only");
            let allow_sorry_free = arg_flag(rest, "--allow-sorry-free");
            let bundle_dir = arg_value(rest, "--bundle-dir").map(PathBuf::from);
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);

            let repo_root =
                plc::find_lean_repo_root(&repo_root).map_err(|e| format!("repo_root: {e}"))?;
            plc::load_dotenv_smart(&repo_root);

            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| format!("failed to build tokio runtime: {e}"))?;

            if focus_line.is_some() && focus_decl.is_some() {
                return Err("use only one of --focus-line or --focus-decl".to_string());
            }

            let dur = StdDuration::from_secs(timeout_s);
            let abs = repo_root.join(&file);
            if !abs.exists() {
                return Err(format!("File not found: {}", abs.display()));
            }
            let original_text = std::fs::read_to_string(&abs)
                .map_err(|e| format!("read {}: {e}", abs.display()))?;

            let has_sorry = plc::locate_sorries_in_text(&original_text, 5, 1)
                .map_err(|e| format!("locate_sorries_in_text failed: {e}"))?
                .len()
                > 0;

            // "Sorry-free" mode: when the target text has no `sorry`/`admit`, we can still obtain a
            // goal dump by synthesizing a shadow decl (`pp_dump; sorry`) using a decl header found
            // near the user's focus.
            let use_shadow_decl = allow_sorry_free && !has_sorry;

            let focus_line_for_goal_dump: Option<usize> = if let Some(line1) = focus_line {
                Some(line1)
            } else if let Some(dn) = focus_decl.as_ref() {
                // Use a light decl-span heuristic from `build_context_pack` to pick a focus line.
                let pack = plc::build_context_pack(&repo_root, &file, Some(dn), None, 0, 0, 0, 80)?;
                Some(pack.focus.start_line)
            } else {
                None
            };

            let focus_decl_effective: Option<String> = if use_shadow_decl {
                if let Some(dn) = focus_decl.as_ref() {
                    Some(dn.trim().to_string())
                } else if let Some(line1) = focus_line_for_goal_dump {
                    plc::nearest_decl_header_in_text(&original_text, line1, 800).map(|d| d.name)
                } else {
                    None
                }
            } else {
                focus_decl.clone()
            };

            if use_shadow_decl && focus_decl_effective.is_none() {
                return Err(
                    "--allow-sorry-free requires --focus-line (near a decl) or --focus-decl"
                        .to_string(),
                );
            }

            let gd = if let Some(line1) = focus_line_for_goal_dump {
                if use_shadow_decl {
                    let dn = focus_decl_effective
                        .as_deref()
                        .ok_or_else(|| "internal: missing focus_decl_effective".to_string())?;
                    rt.block_on(plc::goal_dump_shadow_decl(&repo_root, &file, dn, dur))
                        .map_err(|e| format!("goal_dump_shadow_decl failed: {e}"))?
                } else {
                    rt.block_on(plc::goal_dump_in_text_at(
                        &repo_root,
                        &file,
                        &original_text,
                        dur,
                        Some(line1),
                        None,
                    ))
                    .map_err(|e| format!("goal_dump_in_text_at failed: {e}"))?
                }
            } else {
                // If no focus is provided, we usually need a real `sorry`/`admit` to patch.
                if use_shadow_decl {
                    let dn = focus_decl_effective
                        .as_deref()
                        .ok_or_else(|| "internal: missing focus_decl_effective".to_string())?;
                    rt.block_on(plc::goal_dump_shadow_decl(&repo_root, &file, dn, dur))
                        .map_err(|e| format!("goal_dump_shadow_decl failed: {e}"))?
                } else {
                    rt.block_on(plc::goal_dump_nearest(&repo_root, &file, dur))
                        .map_err(|e| format!("goal_dump_nearest failed: {e}"))?
                }
            };

            let pp = gd
                .get("pp_dump")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let out = if pp_dump_only {
                pp.clone()
            } else {
                json!({
                    "ok": true,
                    "kind": "goal_dump_nearest",
                    "repo_root": repo_root.display().to_string(),
                    "file": file,
                    "focus_line": focus_line_for_goal_dump,
                    "focus_decl": focus_decl,
                    "focus_decl_effective": focus_decl_effective,
                    "timeout_s": timeout_s,
                    "pp_dump": pp,
                    "goal_dump": gd,
                })
            };

            if let Some(dir) = bundle_dir.as_ref() {
                std::fs::create_dir_all(dir)
                    .map_err(|e| format!("failed to create dir {}: {e}", dir.display()))?;
                let goal_dump_path = dir.join("goal_dump.json");
                let pp_dump_path = dir.join("pp_dump.json");
                let manifest_path = dir.join("manifest.json");
                let shadow_path = dir.join("shadow.lean");

                // Always write both shapes to disk (even if `--pp-dump-only` was used).
                let full = json!({
                    "ok": true,
                    "kind": "goal_dump_nearest",
                    "repo_root": repo_root.display().to_string(),
                    "file": file,
                    "focus_line": focus_line_for_goal_dump,
                    "focus_decl": focus_decl,
                    "focus_decl_effective": focus_decl_effective,
                    "timeout_s": timeout_s,
                    "pp_dump": pp,
                    "goal_dump": gd,
                });
                write_json(&goal_dump_path, &full)?;
                write_json(&pp_dump_path, &pp)?;

                // Best-effort: include a `shadow.lean` in the capsule so `goal-try` can run.
                // Prefer explicit focus decl, else use the selected sorry's decl name.
                let mut shadow_written = false;
                let dn = focus_decl_effective.clone().or_else(|| {
                    gd.get("selected_sorry")
                        .and_then(|s| s.get("decl_name"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                });
                if let Some(dn) = dn.as_deref() {
                    if let Ok(txt) = plc::synthesize_pp_dump_shadow_decl(&repo_root, &file, dn) {
                        if std::fs::write(&shadow_path, txt.as_bytes()).is_ok() {
                            shadow_written = true;
                        }
                    }
                }

                let file_bytes = original_text.as_bytes();
                let pp_bytes = serde_json::to_vec(&pp)
                    .map_err(|e| format!("failed to encode pp_dump for hashing: {e}"))?;
                let goals_n = pp
                    .get("goals")
                    .and_then(|g| g.as_array())
                    .map(|g| g.len())
                    .unwrap_or(0);
                let manifest = json!({
                    "kind": "proofpatch_goal_capsule",
                    "tool": { "proofpatch_cli_version": env!("CARGO_PKG_VERSION") },
                    "target": {
                        "repo_root": repo_root.display().to_string(),
                        "file": file,
                        "focus_decl": focus_decl,
                        "focus_decl_effective": focus_decl_effective,
                        "focus_line": focus_line_for_goal_dump,
                    },
                    "mode": {
                        "allow_sorry_free": allow_sorry_free,
                        "used_shadow_decl": use_shadow_decl,
                    },
                    "hashes": {
                        "lean_file_sha256": sha256_hex(file_bytes),
                        "pp_dump_sha256": sha256_hex(&pp_bytes),
                    },
                    "pp_dump": { "goals": goals_n },
                    "paths": {
                        "goal_dump": goal_dump_path.display().to_string(),
                        "pp_dump": pp_dump_path.display().to_string(),
                        "shadow_lean": if shadow_written { json!(shadow_path.display().to_string()) } else { serde_json::Value::Null },
                    }
                });
                write_json(&manifest_path, &manifest)?;
            }

            if let Some(p) = output_json {
                write_json(&p, &out)?;
                println!(
                    "{}",
                    json!({
                        "ok": true,
                        "written": p.display().to_string(),
                        "kind": "goal_dump_nearest",
                        "result_kind": serde_json::Value::Null,
                    })
                    .to_string()
                );
            } else {
                println!("{}", out.to_string());
            }
            Ok(())
        }

        "goal-analyze" => {
            let input_json = arg_value(rest, "--input-json")
                .ok_or_else(|| "missing --input-json".to_string())?;
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);

            let input_label = input_json.clone();
            let input_text = if input_json == "-" {
                let mut s = String::new();
                std::io::stdin()
                    .read_to_string(&mut s)
                    .map_err(|e| format!("read stdin: {e}"))?;
                s
            } else {
                let p = PathBuf::from(&input_json);
                std::fs::read_to_string(&p).map_err(|e| format!("read {}: {e}", p.display()))?
            };
            let v = serde_json::from_str::<serde_json::Value>(&input_text)
                .map_err(|e| format!("json parse {input_label}: {e}"))?;

            // Accept the same shapes as `smt-repro`.
            let pp_dump = if v.get("goals").and_then(|x| x.as_array()).is_some() {
                v.clone()
            } else if let Some(pp) = v.get("pp_dump") {
                pp.clone()
            } else if let Some(pp) = v.get("goal_dump").and_then(|gd| gd.get("pp_dump")) {
                pp.clone()
            } else {
                return Err(
                    "input json must contain a `pp_dump` (or be a `tree-search-nearest` output with `goal_dump.pp_dump`)".to_string(),
                );
            };

            let out = plc::analyze_pp_dump(&pp_dump);
            if let Some(p) = output_json {
                write_json(&p, &out)?;
                println!(
                    "{}",
                    json!({
                        "ok": true,
                        "written": p.display().to_string(),
                        "kind": "goal_analyze",
                        "result_kind": out.get("kind").cloned().unwrap_or(serde_json::Value::Null),
                    })
                    .to_string()
                );
            } else {
                println!("{}", out.to_string());
            }
            Ok(())
        }

        "goal-try" => {
            let capsule_dir = arg_value(rest, "--capsule-dir")
                .ok_or_else(|| "missing --capsule-dir".to_string())
                .map(PathBuf::from)?;
            let timeout_s = arg_u64(rest, "--timeout-s").unwrap_or(20).clamp(1, 600);
            let top_k = arg_u64(rest, "--top-k").unwrap_or(8).clamp(1, 32) as usize;
            let rounds = arg_u64(rest, "--rounds").unwrap_or(2).clamp(1, 10) as usize;
            let beam = arg_u64(rest, "--beam").unwrap_or(1).clamp(1, 8) as usize;
            let with_try_this = arg_flag(rest, "--with-try-this");
            let write_best = arg_flag(rest, "--write-best");
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);

            let pp_dump_path = capsule_dir.join("pp_dump.json");
            let shadow_path = capsule_dir.join("shadow.lean");
            let manifest_path = capsule_dir.join("manifest.json");
            if !pp_dump_path.exists() {
                return Err(format!(
                    "missing pp_dump.json in capsule dir: {}",
                    pp_dump_path.display()
                ));
            }
            if !manifest_path.exists() {
                return Err(format!(
                    "missing manifest.json in capsule dir: {}",
                    manifest_path.display()
                ));
            }

            let pp_text = std::fs::read_to_string(&pp_dump_path)
                .map_err(|e| format!("read {}: {e}", pp_dump_path.display()))?;
            let pp_dump = serde_json::from_str::<serde_json::Value>(&pp_text)
                .map_err(|e| format!("json parse {}: {e}", pp_dump_path.display()))?;

            let manifest_text = std::fs::read_to_string(&manifest_path)
                .map_err(|e| format!("read {}: {e}", manifest_path.display()))?;
            let manifest = serde_json::from_str::<serde_json::Value>(&manifest_text)
                .map_err(|e| format!("json parse {}: {e}", manifest_path.display()))?;
            let repo_root_s = manifest
                .get("target")
                .and_then(|t| t.get("repo_root"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| "manifest.json missing target.repo_root".to_string())?;
            let repo_root = PathBuf::from(repo_root_s);
            let repo_root =
                plc::find_lean_repo_root(&repo_root).map_err(|e| format!("repo_root: {e}"))?;
            plc::load_dotenv_smart(&repo_root);

            // If the capsule is missing `shadow.lean`, try to synthesize it from goal_dump.json.
            if !shadow_path.exists() {
                let goal_dump_path = capsule_dir.join("goal_dump.json");
                if goal_dump_path.exists() {
                    let gd_text = std::fs::read_to_string(&goal_dump_path)
                        .map_err(|e| format!("read {}: {e}", goal_dump_path.display()))?;
                    let gd = serde_json::from_str::<serde_json::Value>(&gd_text)
                        .map_err(|e| format!("json parse {}: {e}", goal_dump_path.display()))?;

                    let file_rel = manifest
                        .get("target")
                        .and_then(|t| t.get("file"))
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| "manifest.json missing target.file".to_string())?;
                    let dn = manifest
                        .get("target")
                        .and_then(|t| t.get("focus_decl"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                        .or_else(|| {
                            gd.get("goal_dump")
                                .and_then(|x| x.get("selected_sorry"))
                                .and_then(|s| s.get("decl_name"))
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string())
                        })
                        .or_else(|| {
                            gd.get("selected_sorry")
                                .and_then(|s| s.get("decl_name"))
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string())
                        });

                    if let Some(dn) = dn.as_deref() {
                        if let Ok(txt) =
                            plc::synthesize_pp_dump_shadow_decl(&repo_root, file_rel, dn)
                        {
                            let _ = std::fs::write(&shadow_path, txt.as_bytes());
                        }
                    }
                }
            }
            if !shadow_path.exists() {
                return Err(format!(
                    "missing shadow.lean in capsule dir: {}",
                    shadow_path.display()
                ));
            }

            let base_shadow = std::fs::read_to_string(&shadow_path)
                .map_err(|e| format!("read {}: {e}", shadow_path.display()))?;

            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| format!("failed to build tokio runtime: {e}"))?;
            // Score versions:
            // - v1 tuple: [goals, hyps_total, first_goal_pretty_chars]  (legacy; kept for compat)
            // - v2 tuple: [goals, total_pretty_chars, hyps_total]       (used for selection/deltas)
            let score_from_pp_dump = |pp: &serde_json::Value| -> Option<serde_json::Value> {
                let goals_n = pp
                    .get("goals")
                    .and_then(|v| v.as_array())
                    .map(|a| a.len())?;
                let mut hyps_total: usize = 0;
                let mut total_pretty_chars: usize = 0;
                let mut max_goal_pretty_chars: usize = 0;
                if let Some(goals) = pp.get("goals").and_then(|v| v.as_array()) {
                    for g in goals {
                        hyps_total += g
                            .get("hyps")
                            .and_then(|v| v.as_array())
                            .map(|a| a.len())
                            .unwrap_or(0);
                        if let Some(p) = g.get("pretty").and_then(|v| v.as_str()) {
                            let n = p.chars().count();
                            total_pretty_chars += n;
                            max_goal_pretty_chars = max_goal_pretty_chars.max(n);
                        }
                    }
                }
                let first_pretty_chars = pp
                    .get("goals")
                    .and_then(|v| v.as_array())
                    .and_then(|a| a.first())
                    .and_then(|g| g.get("pretty"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.chars().count())
                    .unwrap_or(0usize);
                Some(json!({
                    "version": 2,
                    "tuple": [goals_n as u64, total_pretty_chars as u64, hyps_total as u64],
                    "tuple_v1": [goals_n as u64, hyps_total as u64, first_pretty_chars as u64],
                    "goals": goals_n,
                    "hyps_total": hyps_total,
                    "first_pretty_chars": first_pretty_chars,
                    "total_pretty_chars": total_pretty_chars,
                    "max_goal_pretty_chars": max_goal_pretty_chars,
                }))
            };

            let tuple_from_score = |score: &serde_json::Value| -> Option<Vec<u64>> {
                score
                    .get("tuple")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|x| x.as_u64()).collect::<Vec<u64>>())
                    .filter(|v| v.len() == 3)
            };

            let delta_from_scores =
                |baseline: &serde_json::Value, score: &serde_json::Value| -> serde_json::Value {
                    let Some(b) = baseline.get("tuple").and_then(|v| v.as_array()) else {
                        return serde_json::Value::Null;
                    };
                    let Some(s) = score.get("tuple").and_then(|v| v.as_array()) else {
                        return serde_json::Value::Null;
                    };
                    if b.len() != 3 || s.len() != 3 {
                        return serde_json::Value::Null;
                    }
                    let mut d: Vec<i64> = Vec::new();
                    for i in 0..3usize {
                        let bi = b[i]
                            .as_i64()
                            .or_else(|| b[i].as_u64().map(|u| u as i64))
                            .unwrap_or(0);
                        let si = s[i]
                            .as_i64()
                            .or_else(|| s[i].as_u64().map(|u| u as i64))
                            .unwrap_or(0);
                        d.push(si - bi);
                    }
                    serde_json::Value::Array(d.into_iter().map(serde_json::Value::from).collect())
                };

            let mut rounds_out: Vec<serde_json::Value> = Vec::new();
            let mut best_solved_overall: Option<serde_json::Value> = None;
            let mut best_progress_overall: serde_json::Value = serde_json::Value::Null;
            let mut best_shadow_text: Option<String> = None;
            let mut best_written: serde_json::Value = serde_json::Value::Null;
            let mut attempts_total: usize = 0;
            let mut initial_baseline_score: serde_json::Value = serde_json::Value::Null;
            // Cache expensive verifications across rounds/candidates.
            //
            // Keyed by (shadow_sha, kind, candidate_sha) to avoid rerunning `lake env lean` for
            // duplicates across rounds.
            let mut cache_solve: std::collections::HashMap<String, serde_json::Value> =
                std::collections::HashMap::new();
            let mut cache_progress: std::collections::HashMap<String, serde_json::Value> =
                std::collections::HashMap::new();

            // Beam search frontier: each state is a shadow text plus an optional hint score.
            let mut frontier: Vec<(String, Option<Vec<u64>>)> = vec![(base_shadow.clone(), None)];

            for round_i in 0..rounds {
                let mut next_frontier: Vec<(String, Vec<u64>, serde_json::Value)> = Vec::new(); // (shadow, hint_tuple, best_progress_row)
                for (state_idx, (current_shadow, state_hint)) in frontier.iter().enumerate() {
                    let shadow_sha = sha256_hex(current_shadow.as_bytes());
                    // Baseline pp_dump + score for the current hole.
                    let baseline_patch = plc::patch_first_sorry_in_region(
                        &current_shadow,
                        1,
                        999_999,
                        "pp_dump\nsorry",
                    )
                    .map_err(|e| format!("baseline patch failed: {e}"))?;
                    let baseline_txt = baseline_patch.text.clone();
                    let v = rt
                        .block_on(plc::verify_lean_text(
                            &repo_root,
                            &baseline_txt,
                            StdDuration::from_secs(timeout_s),
                        ))
                        .map_err(|e| format!("baseline verify_lean_text failed: {e}"))?;
                    let merged = v.stdout.clone() + "\n" + &v.stderr;
                    let mut baseline_pp_dump = plc::extract_pp_dump_from_lean_output(&merged)
                        .unwrap_or(serde_json::Value::Null);
                    let mut baseline_score =
                        score_from_pp_dump(&baseline_pp_dump).unwrap_or(serde_json::Value::Null);
                    if round_i == 0 {
                        initial_baseline_score = baseline_score.clone();
                    }

                    // Derive candidates from the *current* pp_dump snapshot.
                    let analysis_src = if baseline_pp_dump.is_null() {
                        pp_dump.clone()
                    } else {
                        baseline_pp_dump.clone()
                    };
                    let analysis = plc::analyze_pp_dump(&analysis_src);
                    let mut tactics = analysis
                        .get("tactics")
                        .and_then(|v| v.as_array())
                        .cloned()
                        .unwrap_or_default()
                        .into_iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .take(top_k)
                        .collect::<Vec<_>>();

                    // Also augment candidates by heuristics derived from the current pretty goal state.
                    if let Some(goal_pretty) = baseline_pp_dump
                        .get("goals")
                        .and_then(|v| v.as_array())
                        .and_then(|a| a.first())
                        .and_then(|g| g.get("pretty"))
                        .and_then(|v| v.as_str())
                    {
                        let hint_rules = load_hint_rules(&repo_root, None);
                        tactics.extend(plc::derive_candidates_from_goal_pretty_with_hint_rules(
                            goal_pretty,
                            &hint_rules,
                        ));
                    }

                    // Optional: ask mathlib suggestion tactics for more candidates (bounded).
                    // Use proofpatch-core's implementation so we inherit its heuristics and robustness.
                    let mut try_this_suggestions: Vec<String> = Vec::new();
                    if with_try_this {
                        // The oracle needs a focus line to offer useful `Try this:` suggestions.
                        // Use the same hole selection as our patcher (more robust than scanning).
                        let focus_line_1 = Some(baseline_patch.line);
                        if let Ok(v) = rt.block_on(plc::lean_suggest_in_text_at(
                            &repo_root,
                            "shadow.lean",
                            &current_shadow,
                            StdDuration::from_secs(timeout_s),
                            /* focus_line_1 */ focus_line_1,
                            /* first_error_line_1 */ None,
                        )) {
                            if let Some(arr) = v.get("suggestions").and_then(|x| x.as_array()) {
                                try_this_suggestions = arr
                                    .iter()
                                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                                    .collect();
                            }
                            // If our baseline pp_dump extraction failed, accept the oracle pp_dump.
                            if baseline_pp_dump.is_null() {
                                if let Some(pp) = v.get("pp_dump") {
                                    if pp.is_object() {
                                        baseline_pp_dump = pp.clone();
                                        baseline_score = score_from_pp_dump(&baseline_pp_dump)
                                            .unwrap_or(baseline_score);
                                        if round_i == 0 {
                                            initial_baseline_score = baseline_score.clone();
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let try_this_count = try_this_suggestions.len();

                    // Merge, dedup, bound.
                    tactics.extend(try_this_suggestions.into_iter());
                    {
                        let mut seen = std::collections::HashSet::new();
                        tactics.retain(|s| seen.insert(s.clone()));
                        tactics.truncate(top_k.max(8).min(32));
                    }

                    // Prepare candidate pairs (base, close).
                    let candidates: Vec<(String, String)> = tactics
                        .iter()
                        .filter_map(|c| {
                            // Normalize `by\n  ...` blocks into tactic scripts robustly:
                            // - strip the leading `by\n`
                            // - then drop one common indentation level (`  `) from all lines
                            // This mirrors the tree-search "tactic hole" adapter and prevents
                            // indentation bugs when multi-line candidates are inserted into an
                            // already-indented `by` block.
                            let base = if let Some(rest) = c.strip_prefix("by\n") {
                                let mut out: Vec<String> = Vec::new();
                                for ln in rest.lines() {
                                    out.push(ln.strip_prefix("  ").unwrap_or(ln).to_string());
                                }
                                out.join("\n").trim().to_string()
                            } else if let Some(rest) = c.strip_prefix("by ") {
                                rest.trim().to_string()
                            } else {
                                c.trim().to_string()
                            };
                            if base.is_empty() {
                                return None;
                            }
                            let close = if !base.contains('\n') && !base.contains("done") {
                                format!("{base}; done")
                            } else {
                                base.clone()
                            };
                            Some((base, close))
                        })
                        .collect();

                    let mut results: Vec<serde_json::Value> = Vec::new();
                    for (idx, (cand_base, cand_close)) in candidates.iter().enumerate() {
                        let cand_close_sha = sha256_hex(cand_close.as_bytes());
                        let cand_base_sha = sha256_hex(cand_base.as_bytes());

                        // Solve attempt: close-or-fail candidate.
                        let solve_cache_key = format!("{shadow_sha}::solve::{cand_close_sha}");
                        let solved: bool;
                        let solve_summary: serde_json::Value;
                        let first_error: serde_json::Value;
                        let mut solve_cached = false;
                        if let Some(cached) = cache_solve.get(&solve_cache_key) {
                            solve_cached = true;
                            solved = cached.get("solved").and_then(|v| v.as_bool()) == Some(true);
                            solve_summary = cached
                                .get("summary")
                                .cloned()
                                .unwrap_or(serde_json::Value::Null);
                            first_error = cached
                                .get("first_error")
                                .cloned()
                                .unwrap_or(serde_json::Value::Null);
                        } else {
                            let patched_solve = match plc::patch_first_sorry_in_region(
                                &current_shadow,
                                1,
                                999_999,
                                cand_close,
                            ) {
                                Ok(p) => p.text,
                                Err(e) => {
                                    results.push(json!({
                                        "idx": idx,
                                        "candidate": { "base": cand_base, "close": cand_close },
                                        "ok": false,
                                        "error": format!("patch failed: {e}"),
                                    }));
                                    continue;
                                }
                            };
                            let verify = rt
                                .block_on(plc::verify_lean_text(
                                    &repo_root,
                                    &patched_solve,
                                    StdDuration::from_secs(timeout_s),
                                ))
                                .map_err(|e| format!("verify_lean_text failed: {e}"))?;

                            // `lean` returns exit 0 even with `sorry` (it is a warning),
                            // so "solved" must mean sorry-free as well as error-free.
                            let merged = verify.stdout.clone() + "\n" + &verify.stderr;
                            let has_sorry_warning = merged.contains("declaration uses 'sorry'")
                                || merged.contains("declaration uses 'admit'");
                            solved = verify.ok && !has_sorry_warning;
                            solve_summary = json!({
                                "ok": verify.ok,
                                "timeout": verify.timeout,
                                "returncode": verify.returncode,
                                "errors": verify.stdout.matches(": error:").count()
                                    + verify.stdout.matches(": error(").count()
                                    + verify.stderr.matches(": error:").count()
                                    + verify.stderr.matches(": error(").count(),
                            });
                            let fe = verify
                                .stdout
                                .lines()
                                .find(|l| l.contains(": error:") || l.contains(": error("))
                                .map(|s| s.to_string())
                                .or_else(|| {
                                    verify
                                        .stderr
                                        .lines()
                                        .find(|l| l.contains(": error:") || l.contains(": error("))
                                        .map(|s| s.to_string())
                                });
                            first_error = fe
                                .map(serde_json::Value::String)
                                .unwrap_or(serde_json::Value::Null);

                            cache_solve.insert(
                                solve_cache_key.clone(),
                                json!({
                                    "solved": solved,
                                    "summary": solve_summary,
                                    "first_error": first_error,
                                }),
                            );
                        }

                        // Progress attempt: run candidate, then pp_dump, then sorry.
                        let mut progress: serde_json::Value = serde_json::Value::Null;
                        if !solved {
                            let progress_cache_key =
                                format!("{shadow_sha}::progress::{cand_base_sha}");
                            if let Some(cached) = cache_progress.get(&progress_cache_key) {
                                progress = cached.clone();
                            } else {
                                let mut repl = String::new();
                                repl.push_str(cand_base.trim());
                                if !repl.ends_with('\n') {
                                    repl.push('\n');
                                }
                                repl.push_str("pp_dump\nsorry");
                                let patched_progress = plc::patch_first_sorry_in_region(
                                    &current_shadow,
                                    1,
                                    999_999,
                                    &repl,
                                )
                                .map(|p| p.text);

                                let verify_progress = if let Ok(txt) = patched_progress {
                                    rt.block_on(plc::verify_lean_text(
                                        &repo_root,
                                        &txt,
                                        StdDuration::from_secs(timeout_s),
                                    ))
                                    .ok()
                                } else {
                                    None
                                };

                                let pp_dump = verify_progress.as_ref().and_then(|v| {
                                    plc::extract_pp_dump_from_lean_output(
                                        &(v.stdout.clone() + "\n" + &v.stderr),
                                    )
                                });

                                let score = pp_dump
                                    .as_ref()
                                    .and_then(|pp| score_from_pp_dump(pp))
                                    .unwrap_or(serde_json::Value::Null);
                                let delta = if score.is_null() {
                                    serde_json::Value::Null
                                } else {
                                    delta_from_scores(&baseline_score, &score)
                                };
                                let improves = match (
                                    tuple_from_score(&baseline_score),
                                    tuple_from_score(&score),
                                ) {
                                    (Some(b), Some(s)) => Some(s < b),
                                    _ => None,
                                };

                                let progress_summary = verify_progress.as_ref().map(|v| {
                                    json!({
                                        "ok": v.ok,
                                        "timeout": v.timeout,
                                        "returncode": v.returncode,
                                    })
                                });
                                progress = json!({
                                    "attempted": true,
                                    "baseline_score": baseline_score,
                                    "verify": progress_summary.unwrap_or(serde_json::Value::Null),
                                    "pp_dump_found": pp_dump.is_some(),
                                    "score": score,
                                    "delta": delta,
                                    "improves": improves,
                                });
                                cache_progress.insert(progress_cache_key, progress.clone());
                            }
                        }

                        results.push(json!({
                        "idx": idx,
                        "candidate": { "base": cand_base, "close": cand_close },
                        "solve": { "verify": solve_summary, "first_error": first_error },
                        "progress": progress,
                        "cache": { "solve": solve_cached, "progress": progress.get("attempted").is_some() },
                    }));

                        if solved {
                            break;
                        }
                    }

                    attempts_total += results.len();

                    let best_solved = results
                        .iter()
                        .find(|r| {
                            r.get("solve")
                                .and_then(|v| v.get("verify"))
                                .and_then(|v| v.get("ok"))
                                .and_then(|b| b.as_bool())
                                == Some(true)
                        })
                        .cloned();

                    let best_progress = if best_solved.is_some() {
                        serde_json::Value::Null
                    } else {
                        // Pick minimum lexicographically by our score tuple, preferring improves=true.
                        let mut best: Option<(Vec<u64>, serde_json::Value)> = None;
                        for r in &results {
                            let t = r
                                .get("progress")
                                .and_then(|p| p.get("score"))
                                .and_then(|s| tuple_from_score(s))
                                .filter(|v| v.len() == 3);
                            let Some(tuple) = t else { continue };
                            let improves = r
                                .get("progress")
                                .and_then(|p| p.get("improves"))
                                .and_then(|v| v.as_bool());
                            match &best {
                                None => best = Some((tuple, r.clone())),
                                Some((best_tuple, best_r)) => {
                                    let best_improves = best_r
                                        .get("progress")
                                        .and_then(|p| p.get("improves"))
                                        .and_then(|v| v.as_bool());
                                    let better = match (improves, best_improves) {
                                        (Some(true), Some(false) | None) => true,
                                        (Some(false) | None, Some(true)) => false,
                                        _ => tuple < *best_tuple,
                                    };
                                    if better {
                                        best = Some((tuple, r.clone()));
                                    }
                                }
                            }
                        }
                        best.map(|(_, r)| r).unwrap_or(serde_json::Value::Null)
                    };

                    rounds_out.push(json!({
                    "round": round_i + 1,
                    "state_idx": state_idx,
                    "state_hint": state_hint.clone(),
                    "analysis": analysis,
                    "baseline": { "pp_dump": baseline_pp_dump, "score": baseline_score },
                    "try_this": { "enabled": with_try_this, "count": try_this_count },
                    "attempts": results.len(),
                    "results": results,
                    "best": { "solved": best_solved.clone().unwrap_or(serde_json::Value::Null), "progress": best_progress.clone() },
                }));

                    if let Some(sol) = best_solved {
                        best_solved_overall = Some(sol.clone());
                        // Write the solved text.
                        let cand_close = sol
                            .get("candidate")
                            .and_then(|c| c.get("close"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let solved_text = plc::patch_first_sorry_in_region(
                            current_shadow,
                            1,
                            999_999,
                            cand_close,
                        )
                        .map(|p| p.text)
                        .ok();
                        best_shadow_text = solved_text;
                        // Stop everything on first solved (beam search short-circuit).
                        frontier.clear();
                        break;
                    }

                    if !best_progress.is_null() {
                        let improves = best_progress
                            .get("progress")
                            .and_then(|p| p.get("improves"))
                            .and_then(|v| v.as_bool());
                        // Non-regression hill-climb: only apply steps that *actually* improve baseline.
                        if improves != Some(true) {
                            continue;
                        }

                        best_progress_overall = best_progress.clone();
                        let cand_base = best_progress
                            .get("candidate")
                            .and_then(|c| c.get("base"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let repl = format!("{}\n{}", cand_base.trim(), "sorry");
                        let next_shadow =
                            plc::patch_first_sorry_in_region(current_shadow, 1, 999_999, &repl)
                                .map(|p| p.text)
                                .ok();
                        if let Some(ns) = next_shadow {
                            best_shadow_text = Some(ns.clone());
                            // Use the progress score (post-tactic pp_dump) as a hint for ranking the frontier.
                            let hint_tuple = best_progress
                                .get("progress")
                                .and_then(|p| p.get("score"))
                                .and_then(|s| tuple_from_score(s))
                                .unwrap_or_else(|| vec![u64::MAX, u64::MAX, u64::MAX]);
                            next_frontier.push((ns, hint_tuple, best_progress));
                        }
                    }
                } // end per-state loop

                if best_solved_overall.is_some() {
                    break;
                }

                // Dedup next frontier by shadow hash, then keep top `beam` by hint tuple.
                let mut seen_sha: std::collections::HashSet<String> =
                    std::collections::HashSet::new();
                next_frontier.retain(|(txt, _, _)| seen_sha.insert(sha256_hex(txt.as_bytes())));
                next_frontier.sort_by(|a, b| a.1.cmp(&b.1));
                let kept = next_frontier.into_iter().take(beam).collect::<Vec<_>>();

                // Optional artifacts: write the best frontier state per round.
                if write_best {
                    if let Some((txt, _, _)) = kept.first() {
                        let p = capsule_dir.join(format!("shadow.round_{}.lean", round_i + 1));
                        let _ = std::fs::write(&p, txt);
                    }
                }

                frontier = kept
                    .iter()
                    .map(|(txt, hint, _row)| (txt.clone(), Some(hint.clone())))
                    .collect();

                // If we couldn't produce any improving next state, stop.
                if frontier.is_empty() {
                    break;
                }
            }

            let result_kind = if best_solved_overall.is_some() {
                "solved"
            } else if !best_progress_overall.is_null() {
                "progress"
            } else {
                "unsolved"
            };

            if write_best {
                let out_path = capsule_dir.join("shadow.best.lean");
                if let Some(txt) = best_shadow_text.as_ref() {
                    if std::fs::write(&out_path, txt).is_ok() {
                        best_written = serde_json::Value::String(out_path.display().to_string());
                    }
                }
            }

            let out = json!({
                "ok": true,
                "kind": "goal_try",
                "result_kind": result_kind,
                "capsule_dir": capsule_dir.display().to_string(),
                "beam": beam,
                "rounds": { "requested": rounds, "completed": rounds_out.len() },
                "baseline_score": initial_baseline_score,
                "attempts": attempts_total,
                "best": {
                    "solved": best_solved_overall.unwrap_or(serde_json::Value::Null),
                    "progress": best_progress_overall,
                },
                "best_written": best_written,
                "round_results": rounds_out,
            });

            if let Some(p) = output_json {
                write_json(&p, &out)?;
                println!(
                    "{}",
                    json!({
                        "ok": true,
                        "written": p.display().to_string(),
                        "kind": "goal_try",
                        "result_kind": out["result_kind"],
                    })
                    .to_string()
                );
            } else {
                println!("{}", out.to_string());
            }
            Ok(())
        }

        "agent-step" => {
            let repo_root = arg_value(rest, "--repo")
                .ok_or_else(|| "missing --repo".to_string())
                .map(PathBuf::from)?;
            let file = arg_value(rest, "--file").ok_or_else(|| "missing --file".to_string())?;
            let timeout_s = arg_u64(rest, "--timeout-s").unwrap_or(180);
            let write = arg_flag(rest, "--write");
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);

            let repo_root =
                plc::find_lean_repo_root(&repo_root).map_err(|e| format!("repo_root: {e}"))?;
            plc::load_dotenv_smart(&repo_root);
            let abs = repo_root.join(&file);
            if !abs.exists() {
                return Err(format!("File not found: {}", abs.display()));
            }
            let original_text = std::fs::read_to_string(&abs)
                .map_err(|e| format!("read {}: {e}", abs.display()))?;

            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| format!("failed to build tokio runtime: {e}"))?;

            let mut nodes = Vec::new();
            let mut edges = Vec::new();

            let verify0 = rt
                .block_on(plc::verify_lean_file(
                    &repo_root,
                    &file,
                    StdDuration::from_secs(timeout_s),
                ))
                .map_err(|e| format!("verify failed: {e}"))?;
            let first_error_loc = plc::parse_first_error_loc(&verify0.stdout, &verify0.stderr);
            let first_error_line = first_error_loc.as_ref().map(|l| l.line);
            let first_error_text = verify0
                .stdout
                .lines()
                .find(|l| l.contains(": error:"))
                .map(|s| s.to_string());

            nodes.push(json!({
                "id": "verify0",
                "kind": "verify",
                "ok": verify0.ok,
                "returncode": verify0.returncode,
                "first_error_loc": first_error_loc,
                "first_error": first_error_text,
            }));

            if verify0.ok {
                let out = json!({
                    "ok": true,
                    "repo_root": repo_root.display().to_string(),
                    "file": file,
                    "dag": { "nodes": nodes, "edges": edges },
                    "note": "already OK; no action executed",
                });
                println!("{}", out.to_string());
                return Ok(());
            }

            // Execute `next_action` heuristically: fix-first-error with safe mechanical edits.
            let (patched_text, edits) = apply_mechanical_fixes_for_first_error(
                &original_text,
                first_error_line,
                first_error_text.as_deref(),
            );

            nodes.push(json!({
                "id": "mech_fix1",
                "kind": "mechanical_fix",
                "applied": !edits.is_empty(),
                "edits": edits,
                "write": write,
            }));
            edges.push(json!({ "from": "verify0", "to": "mech_fix1" }));

            let mut wrote_path: Option<String> = None;
            if write && !edits.is_empty() {
                std::fs::write(&abs, patched_text.as_bytes())
                    .map_err(|e| format!("write {}: {e}", abs.display()))?;
                wrote_path = Some(abs.display().to_string());
            }

            let verify1 = if write && !edits.is_empty() {
                rt.block_on(plc::verify_lean_file(
                    &repo_root,
                    &file,
                    StdDuration::from_secs(timeout_s),
                ))
                .map_err(|e| format!("verify failed: {e}"))?
            } else {
                rt.block_on(plc::verify_lean_text(
                    &repo_root,
                    &patched_text,
                    StdDuration::from_secs(timeout_s),
                ))
                .map_err(|e| format!("verify failed: {e}"))?
            };

            nodes.push(json!({
                "id": "verify1",
                "kind": "verify",
                "ok": verify1.ok,
                "returncode": verify1.returncode,
                "first_error_loc": plc::parse_first_error_loc(&verify1.stdout, &verify1.stderr),
                "first_error": verify1.stdout.lines().find(|l| l.contains(": error:")),
            }));
            edges.push(json!({ "from": "mech_fix1", "to": "verify1" }));

            let full = json!({
                "ok": true,
                "repo_root": repo_root.display().to_string(),
                "file": file,
                "written_file": wrote_path,
                "dag": { "nodes": nodes, "edges": edges },
            });

            if let Some(p) = output_json {
                write_json(&p, &full)?;
                let small = json!({
                    "ok": true,
                    "written": p.display().to_string(),
                    "kind": "agent_step",
                    "result_kind": serde_json::Value::Null,
                    "file": file,
                    "written_file": wrote_path,
                    "verify1_ok": verify1.ok,
                });
                println!("{}", small.to_string());
            } else {
                println!("{}", full.to_string());
            }
            Ok(())
        }

        "prompt" => {
            let repo_root = arg_value(rest, "--repo")
                .ok_or_else(|| "missing --repo".to_string())
                .map(PathBuf::from)?;
            let file = arg_value(rest, "--file").ok_or_else(|| "missing --file".to_string())?;
            let lemma = arg_value(rest, "--lemma").ok_or_else(|| "missing --lemma".to_string())?;
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);

            let repo_root =
                plc::find_lean_repo_root(&repo_root).map_err(|e| format!("repo_root: {e}"))?;
            plc::load_dotenv_smart(&repo_root);

            let payload = plc::build_proof_prompt(&repo_root, &file, &lemma)?;
            let out = serde_json::to_value(payload).map_err(|e| format!("json encode: {e}"))?;

            if let Some(p) = output_json {
                write_json(&p, &out)?;
                println!(
                    "{}",
                    json!({
                        "ok": true,
                        "written": p.display().to_string(),
                        "kind": out["kind"],
                        "result_kind": out["result_kind"],
                    })
                    .to_string()
                );
            } else {
                println!("{}", out.to_string());
            }
            Ok(())
        }

        "rubberduck-prompt" => {
            let repo_root = arg_value(rest, "--repo")
                .ok_or_else(|| "missing --repo".to_string())
                .map(PathBuf::from)?;
            let file = arg_value(rest, "--file").ok_or_else(|| "missing --file".to_string())?;
            let lemma = arg_value(rest, "--lemma").ok_or_else(|| "missing --lemma".to_string())?;
            let diagnostics_file = arg_value(rest, "--diagnostics-file").map(PathBuf::from);
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);

            let repo_root =
                plc::find_lean_repo_root(&repo_root).map_err(|e| format!("repo_root: {e}"))?;
            plc::load_dotenv_smart(&repo_root);

            let diagnostics = if let Some(p) = diagnostics_file {
                Some(
                    std::fs::read_to_string(&p)
                        .map_err(|e| format!("read {}: {e}", p.display()))?,
                )
            } else {
                None
            };

            let payload =
                plc::build_rubberduck_prompt(&repo_root, &file, &lemma, diagnostics.as_deref())?;
            let out = serde_json::to_value(payload).map_err(|e| format!("json encode: {e}"))?;

            if let Some(p) = output_json {
                write_json(&p, &out)?;
                println!(
                    "{}",
                    json!({
                        "ok": true,
                        "written": p.display().to_string(),
                        "kind": "rubberduck_prompt",
                        "result_kind": serde_json::Value::Null,
                    })
                    .to_string()
                );
            } else {
                println!("{}", out.to_string());
            }
            Ok(())
        }

        "locate-sorries" => {
            let repo_root = arg_value(rest, "--repo")
                .ok_or_else(|| "missing --repo".to_string())
                .map(PathBuf::from)?;
            let file = arg_value(rest, "--file").ok_or_else(|| "missing --file".to_string())?;
            let max_sorries = arg_u64(rest, "--max-sorries").unwrap_or(50) as usize;
            let context_lines = arg_u64(rest, "--context-lines").unwrap_or(1) as usize;
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);

            let repo_root =
                plc::find_lean_repo_root(&repo_root).map_err(|e| format!("repo_root: {e}"))?;
            let locs = plc::locate_sorries_in_file(&repo_root, &file, max_sorries, context_lines)?;

            let out = json!({
                "repo_root": repo_root.display().to_string(),
                "file": file,
                "count": locs.len(),
                "max_sorries": max_sorries,
                "context_lines": context_lines,
                "locations": locs
            });
            if let Some(p) = output_json {
                write_json(&p, &out)?;
                println!(
                    "{}",
                    json!({
                        "ok": true,
                        "written": p.display().to_string(),
                        "kind": "locate_sorries",
                        "result_kind": serde_json::Value::Null,
                    })
                    .to_string()
                );
            } else {
                println!("{}", out.to_string());
            }
            Ok(())
        }

        "verify-summary" => {
            let repo_root = arg_value(rest, "--repo")
                .ok_or_else(|| "missing --repo".to_string())
                .map(PathBuf::from)?;
            let file = arg_value(rest, "--file").ok_or_else(|| "missing --file".to_string())?;
            let timeout_s = arg_u64(rest, "--timeout-s").unwrap_or(120);
            let include_raw_verify = arg_flag(rest, "--include-raw-verify");
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);

            let repo_root =
                plc::find_lean_repo_root(&repo_root).map_err(|e| format!("repo_root: {e}"))?;
            plc::load_dotenv_smart(&repo_root);

            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| format!("failed to build tokio runtime: {e}"))?;
            let raw = rt
                .block_on(plc::verify_lean_file(
                    &repo_root,
                    &file,
                    StdDuration::from_secs(timeout_s),
                ))
                .map_err(|e| format!("verify failed: {e}"))?;

            let raw_v = serde_json::to_value(raw).map_err(|e| format!("serialize verify: {e}"))?;
            let summary = verify_summary_from_raw_value(&raw_v);

            let out = json!({
                "repo_root": repo_root.display().to_string(),
                "file": file,
                "verify": {
                    "summary": summary,
                    "raw": if include_raw_verify { raw_v } else { serde_json::Value::Null }
                }
            });

            if let Some(p) = output_json {
                write_json(&p, &out)?;
                println!(
                    "{}",
                    json!({
                        "ok": true,
                        "written": p.display().to_string(),
                        "kind": "verify_summary",
                        "result_kind": serde_json::Value::Null,
                    })
                    .to_string()
                );
            } else {
                println!("{}", out.to_string());
            }
            Ok(())
        }

        "patch" => {
            let repo_root = arg_value(rest, "--repo")
                .ok_or_else(|| "missing --repo".to_string())
                .map(PathBuf::from)?;
            let file = arg_value(rest, "--file").ok_or_else(|| "missing --file".to_string())?;
            let lemma = arg_value(rest, "--lemma").ok_or_else(|| "missing --lemma".to_string())?;
            let replacement_file = arg_value(rest, "--replacement-file")
                .ok_or_else(|| "missing --replacement-file".to_string())
                .map(PathBuf::from)?;
            let timeout_s = arg_u64(rest, "--timeout-s").unwrap_or(120);
            let write = arg_flag(rest, "--write");
            let include_raw_verify = arg_flag(rest, "--include-raw-verify");
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);

            let repo_root =
                plc::find_lean_repo_root(&repo_root).map_err(|e| format!("repo_root: {e}"))?;
            plc::load_dotenv_smart(&repo_root);

            let abs = repo_root.join(&file);
            if !abs.exists() {
                return Err(format!("File not found: {}", abs.display()));
            }
            let original_text = std::fs::read_to_string(&abs)
                .map_err(|e| format!("read {}: {e}", abs.display()))?;
            let replacement = std::fs::read_to_string(&replacement_file)
                .map_err(|e| format!("read {}: {e}", replacement_file.display()))?;

            let patched = plc::patch_first_sorry_in_decl(&original_text, &lemma, &replacement)?;
            let still_has_sorry = plc::decl_block_contains_sorry(&patched.text, &lemma)?;

            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| format!("failed to build tokio runtime: {e}"))?;
            let mut written_file: Option<String> = None;
            if write {
                std::fs::write(&abs, patched.text.as_bytes())
                    .map_err(|e| format!("write {}: {e}", abs.display()))?;
                written_file = Some(abs.display().to_string());
            }
            let verify = if write {
                rt.block_on(plc::verify_lean_file(
                    &repo_root,
                    &file,
                    StdDuration::from_secs(timeout_s),
                ))
                .map_err(|e| format!("verify failed: {e}"))?
            } else {
                rt.block_on(plc::verify_lean_text(
                    &repo_root,
                    &patched.text,
                    StdDuration::from_secs(timeout_s),
                ))
                .map_err(|e| format!("verify failed: {e}"))?
            };
            let verify_raw_v =
                serde_json::to_value(verify).map_err(|e| format!("serialize verify: {e}"))?;
            let verify_summary = verify_summary_from_raw_value(&verify_raw_v);

            let out = json!({
                "repo_root": repo_root.display().to_string(),
                "file": file,
                "lemma": lemma,
                "written_file": written_file,
                "patch": {
                    "line": patched.line,
                    "before": patched.before,
                    "after": patched.after,
                    "indent": patched.indent,
                },
                "lemma_still_contains_sorry": still_has_sorry,
                "verify": {
                    "summary": verify_summary,
                    "raw": if include_raw_verify { verify_raw_v } else { serde_json::Value::Null }
                },
            });

            if let Some(p) = output_json {
                write_json(&p, &out)?;
                println!(
                    "{}",
                    json!({
                        "ok": true,
                        "written": p.display().to_string(),
                        "kind": "patch",
                        "result_kind": serde_json::Value::Null,
                    })
                    .to_string()
                );
            } else {
                println!("{}", out.to_string());
            }
            Ok(())
        }

        "patch-region" => {
            let repo_root = arg_value(rest, "--repo")
                .ok_or_else(|| "missing --repo".to_string())
                .map(PathBuf::from)?;
            let file = arg_value(rest, "--file").ok_or_else(|| "missing --file".to_string())?;
            let start_line = arg_u64(rest, "--start-line")
                .ok_or_else(|| "missing --start-line".to_string())?
                as usize;
            let end_line = arg_u64(rest, "--end-line")
                .ok_or_else(|| "missing --end-line".to_string())?
                as usize;
            let replacement_file = arg_value(rest, "--replacement-file")
                .ok_or_else(|| "missing --replacement-file".to_string())
                .map(PathBuf::from)?;
            let timeout_s = arg_u64(rest, "--timeout-s").unwrap_or(120);
            let write = arg_flag(rest, "--write");
            let include_raw_verify = arg_flag(rest, "--include-raw-verify");
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);

            let repo_root =
                plc::find_lean_repo_root(&repo_root).map_err(|e| format!("repo_root: {e}"))?;
            plc::load_dotenv_smart(&repo_root);

            let abs = repo_root.join(&file);
            if !abs.exists() {
                return Err(format!("File not found: {}", abs.display()));
            }
            let original_text = std::fs::read_to_string(&abs)
                .map_err(|e| format!("read {}: {e}", abs.display()))?;
            let replacement = std::fs::read_to_string(&replacement_file)
                .map_err(|e| format!("read {}: {e}", replacement_file.display()))?;

            let patched = plc::patch_first_sorry_in_region(
                &original_text,
                start_line,
                end_line,
                &replacement,
            )?;

            // “still contains sorry” is scoped to the patched region (post-patch).
            let region_text = patched
                .text
                .lines()
                .skip(start_line.saturating_sub(1))
                .take(end_line.saturating_sub(start_line).saturating_add(1))
                .collect::<Vec<_>>()
                .join("\n");
            let region_still_contains_sorry = region_text.contains("sorry");

            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| format!("failed to build tokio runtime: {e}"))?;

            let mut written_file: Option<String> = None;
            if write {
                std::fs::write(&abs, patched.text.as_bytes())
                    .map_err(|e| format!("write {}: {e}", abs.display()))?;
                written_file = Some(abs.display().to_string());
            }
            let verify = if write {
                rt.block_on(plc::verify_lean_file(
                    &repo_root,
                    &file,
                    StdDuration::from_secs(timeout_s),
                ))
                .map_err(|e| format!("verify failed: {e}"))?
            } else {
                rt.block_on(plc::verify_lean_text(
                    &repo_root,
                    &patched.text,
                    StdDuration::from_secs(timeout_s),
                ))
                .map_err(|e| format!("verify failed: {e}"))?
            };
            let verify_raw_v =
                serde_json::to_value(verify).map_err(|e| format!("serialize verify: {e}"))?;
            let verify_summary = verify_summary_from_raw_value(&verify_raw_v);

            let out = json!({
                "repo_root": repo_root.display().to_string(),
                "file": file,
                "written_file": written_file,
                "patch": {
                    "line": patched.line,
                    "before": patched.before,
                    "after": patched.after,
                    "indent": patched.indent,
                },
                "region": {
                    "start_line": start_line,
                    "end_line": end_line,
                    "still_contains_sorry": region_still_contains_sorry,
                },
                "verify": {
                    "summary": verify_summary,
                    "raw": if include_raw_verify { verify_raw_v } else { serde_json::Value::Null }
                },
            });

            if let Some(p) = output_json {
                write_json(&p, &out)?;
                println!(
                    "{}",
                    json!({
                        "ok": true,
                        "written": p.display().to_string(),
                        "kind": "patch_region",
                        "result_kind": serde_json::Value::Null,
                    })
                    .to_string()
                );
            } else {
                println!("{}", out.to_string());
            }
            Ok(())
        }

        "patch-nearest" => {
            let repo_root = arg_value(rest, "--repo")
                .ok_or_else(|| "missing --repo".to_string())
                .map(PathBuf::from)?;
            let file = arg_value(rest, "--file").ok_or_else(|| "missing --file".to_string())?;
            let replacement_file = arg_value(rest, "--replacement-file")
                .ok_or_else(|| "missing --replacement-file".to_string())
                .map(PathBuf::from)?;
            let timeout_s = arg_u64(rest, "--timeout-s").unwrap_or(120);
            let write = arg_flag(rest, "--write");
            let include_raw_verify = arg_flag(rest, "--include-raw-verify");
            let max_sorries = arg_u64(rest, "--max-sorries").unwrap_or(50) as usize;
            let context_lines = arg_u64(rest, "--context-lines").unwrap_or(1) as usize;
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);

            let repo_root =
                plc::find_lean_repo_root(&repo_root).map_err(|e| format!("repo_root: {e}"))?;
            plc::load_dotenv_smart(&repo_root);

            let abs = repo_root.join(&file);
            if !abs.exists() {
                return Err(format!("File not found: {}", abs.display()));
            }
            let original_text = std::fs::read_to_string(&abs)
                .map_err(|e| format!("read {}: {e}", abs.display()))?;
            let replacement = std::fs::read_to_string(&replacement_file)
                .map_err(|e| format!("read {}: {e}", replacement_file.display()))?;

            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| format!("failed to build tokio runtime: {e}"))?;

            // Baseline verify (helps choose the `sorry` nearest to the first error, if any).
            let baseline = rt
                .block_on(plc::verify_lean_file(
                    &repo_root,
                    &file,
                    StdDuration::from_secs(timeout_s),
                ))
                .map_err(|e| format!("verify failed: {e}"))?;
            let baseline_raw_v =
                serde_json::to_value(baseline).map_err(|e| format!("serialize verify: {e}"))?;
            let baseline_summary = verify_summary_from_raw_value(&baseline_raw_v);

            // Establish a focus declaration for the whole search (best-effort).
            // This makes depth>1 behave like a proof-tree search: keep patching within the same lemma.
            let _focus_decl_name = {
                let first_error_line_1 = baseline_summary
                    .get("first_error_loc")
                    .and_then(|l| l.get("line"))
                    .and_then(|v| v.as_u64())
                    .map(|x| x as usize);
                let locs0 = plc::locate_sorries_in_text(&original_text, 200, 1).unwrap_or_default();
                plc::select_primary_sorry(first_error_line_1, &locs0)
                    .and_then(|s| s.decl_name.clone())
            };
            let first_error_line_1 = baseline_summary
                .get("first_error_loc")
                .and_then(|v| v.get("line"))
                .and_then(|v| v.as_u64())
                .map(|x| x as usize);

            let locs = plc::locate_sorries_in_text(&original_text, max_sorries, context_lines)?;
            let selected =
                plc::select_primary_sorry(first_error_line_1, &locs).ok_or_else(|| {
                    format!(
                        "No `sorry`/`admit` tokens found (max_sorries={}, context_lines={}).",
                        max_sorries, context_lines
                    )
                })?;

            let patched = plc::patch_first_sorry_in_region(
                &original_text,
                selected.region_start,
                selected.region_end,
                &replacement,
            )?;

            // “still contains sorry” is scoped to the patched region (post-patch).
            let region_text = patched
                .text
                .lines()
                .skip(selected.region_start.saturating_sub(1))
                .take(
                    selected
                        .region_end
                        .saturating_sub(selected.region_start)
                        .saturating_add(1),
                )
                .collect::<Vec<_>>()
                .join("\n");
            let region_still_contains_sorry = region_text.contains("sorry");

            let mut written_file: Option<String> = None;
            if write {
                std::fs::write(&abs, patched.text.as_bytes())
                    .map_err(|e| format!("write {}: {e}", abs.display()))?;
                written_file = Some(abs.display().to_string());
            }
            let verify = if write {
                rt.block_on(plc::verify_lean_file(
                    &repo_root,
                    &file,
                    StdDuration::from_secs(timeout_s),
                ))
                .map_err(|e| format!("verify failed: {e}"))?
            } else {
                rt.block_on(plc::verify_lean_text(
                    &repo_root,
                    &patched.text,
                    StdDuration::from_secs(timeout_s),
                ))
                .map_err(|e| format!("verify failed: {e}"))?
            };
            let verify_raw_v =
                serde_json::to_value(verify).map_err(|e| format!("serialize verify: {e}"))?;
            let verify_summary = verify_summary_from_raw_value(&verify_raw_v);

            let selected_v =
                serde_json::to_value(&selected).map_err(|e| format!("serialize sorry: {e}"))?;

            let out = json!({
                "repo_root": repo_root.display().to_string(),
                "file": file,
                "written_file": written_file,
                "selection": {
                    "first_error_line": first_error_line_1,
                    "selected_sorry": selected_v,
                },
                "patch": {
                    "line": patched.line,
                    "before": patched.before,
                    "after": patched.after,
                    "indent": patched.indent,
                },
                "region": {
                    "start_line": selected.region_start,
                    "end_line": selected.region_end,
                    "still_contains_sorry": region_still_contains_sorry,
                },
                "baseline_verify": { "summary": baseline_summary },
                "verify": {
                    "summary": verify_summary,
                    "raw": if include_raw_verify { verify_raw_v } else { serde_json::Value::Null }
                },
            });

            if let Some(p) = output_json {
                write_json(&p, &out)?;
                println!(
                    "{}",
                    json!({
                        "ok": true,
                        "written": p.display().to_string(),
                        "kind": "patch_nearest",
                        "result_kind": serde_json::Value::Null,
                    })
                    .to_string()
                );
            } else {
                println!("{}", out.to_string());
            }
            Ok(())
        }

        "tree-search-nearest" => {
            #[derive(Clone)]
            struct Node {
                id: usize,
                depth: usize,
                text: String,
                // Best-effort: keep the search focused within one declaration (lemma/theorem/def).
                focus_decl_name: Option<String>,
                // Stronger focus: prefer patching the `sorry` closest to this line.
                focus_line: Option<usize>,
                // Goal-branch stabilization: when we can cheaply identify the goal signature for a hole,
                // prefer continuing to work on the same goal across depths (cache-first; no extra Lean calls).
                focus_goal_sig: Option<u64>,
                // For traceability.
                last_region: Option<(usize, usize)>,
                last_replacement: Option<String>,
                parent_id: Option<usize>,
                // Cached evaluation.
                verify_raw: Option<serde_json::Value>,
                verify_summary: Option<serde_json::Value>,
                sorries: Option<usize>,
                conservative_sorries: Option<usize>,
                // Optional per-hole SMT signal (entails/unknown + source) used for ranking.
                // This is best-effort and should never trigger additional Lean calls.
                smt_hint: Option<serde_json::Value>,
                // Compact explanation of candidate ranking at the expanded hole (top-k).
                rank_hint: Option<serde_json::Value>,
            }

            #[derive(Clone)]
            struct CachedEval {
                len: usize,
                verify_raw: serde_json::Value,
                verify_summary: serde_json::Value,
                sorries: usize,
                conservative_sorries: usize,
            }

            use plc::tree_search::{
                adapt_candidates_for_error, adapt_candidates_for_sorry_context,
                default_det_candidates, extract_initial_goal_block, hash_state_key, hash_text,
                is_made_no_progress, parse_json_string_array, progress_score_key,
                sanitize_candidates, verify_score_key,
            };

            let repo_root = arg_value(rest, "--repo")
                .ok_or_else(|| "missing --repo".to_string())
                .map(PathBuf::from)?;
            let file = arg_value(rest, "--file").ok_or_else(|| "missing --file".to_string())?;
            let timeout_s = arg_u64(rest, "--timeout-s").unwrap_or(120);
            let total_timeout_s = arg_u64(rest, "--total-timeout-s")
                .unwrap_or(timeout_s)
                .max(1);
            let log_level = arg_u64(rest, "--log-level").unwrap_or(1);
            let events_jsonl_requested = arg_value(rest, "--events-jsonl").map(PathBuf::from);
            let mut events_jsonl = events_jsonl_requested.clone();
            let events_keep = arg_u64(rest, "--events-keep")
                .unwrap_or(512)
                .clamp(16, 10_000) as usize;
            let events_all_keep = arg_u64(rest, "--events-all-keep")
                .unwrap_or(5000)
                .clamp(events_keep as u64, 200_000) as usize;
            // Global wall-clock budget: start it as early as possible so it covers repo-root
            // resolution, file IO, and any process/LSP startup we do inside this command.
            let prof_t0 = std::time::Instant::now();
            let run_deadline = std::time::Instant::now()
                .checked_add(StdDuration::from_secs(total_timeout_s))
                .unwrap_or_else(std::time::Instant::now);
            let mut bailed_total_timeout = false;

            let remaining_ms = |deadline: std::time::Instant| -> u64 {
                let now = std::time::Instant::now();
                if now >= deadline {
                    0
                } else {
                    deadline.duration_since(now).as_millis() as u64
                }
            };

            let budget_dur = |cap_s: u64| -> Option<StdDuration> {
                let rem = remaining_ms(run_deadline);
                if rem == 0 {
                    None
                } else {
                    let cap_ms = cap_s.saturating_mul(1000);
                    Some(StdDuration::from_millis(rem.min(cap_ms).max(1)))
                }
            };
            // Goal-dump preflight should be much cheaper than full verify: it is used only for
            // goal-first selection and ranking hints. Keep it bounded regardless of `--timeout-s`.
            let goal_dump_timeout_s = std::env::var("PROOFPATCH_GOAL_DUMP_TIMEOUT_S")
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(12)
                .max(2)
                .min(timeout_s);
            // Lean oracle (`lean_suggest_in_text_at`) should also be bounded: it's a heuristic for candidate
            // generation and can be slow on hard goals. Default is intentionally smaller than `--timeout-s`.
            let oracle_timeout_s = std::env::var("PROOFPATCH_ORACLE_TIMEOUT_S")
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(20)
                .max(2)
                .min(timeout_s);
            let beam = arg_u64(rest, "--beam").unwrap_or(4) as usize;
            let max_nodes = arg_u64(rest, "--max-nodes").unwrap_or(20) as usize;
            let depth = arg_u64(rest, "--depth").unwrap_or(2) as usize;
            // Defaults discipline:
            // - baseline default: `det` (fast, offline)
            // - if a repo-owned research preset is provided, default to `lean-try` (still offline)
            //   and allow LLM escalation once we’re stuck.
            let (candidates_mode, candidates_mode_source) =
                if let Some(v) = arg_value(rest, "--candidates") {
                    (v, "explicit")
                } else if arg_value(rest, "--research-preset").is_some() {
                    ("lean-try".to_string(), "research_preset_default")
                } else {
                    ("det".to_string(), "baseline_default")
                };
            let lean_oracle_per_node = arg_flag(rest, "--lean-oracle-per-node");
            let lean_oracle_max_calls =
                arg_u64(rest, "--lean-oracle-max-calls").unwrap_or(12) as usize;
            let rollout_k = arg_u64(rest, "--rollout-k").unwrap_or(0) as usize;
            let dedup_goal_expansions = arg_flag(rest, "--dedup-goal-expansions");
            // When using `lean-try` or a research preset, goal-first scheduling is usually a win.
            // Keep it bounded and non-surprising: default to 3 only in those modes.
            let goal_first_k_raw = arg_u64(rest, "--goal-first-k").unwrap_or(0) as usize;
            let goal_first_k_explicit = arg_value(rest, "--goal-first-k").is_some();
            let mut goal_first_k_source = if goal_first_k_explicit {
                "explicit"
            } else if candidates_mode == "lean-try"
                || arg_value(rest, "--research-preset").is_some()
            {
                "mode_default"
            } else {
                "default"
            };
            let goal_first_k_defaulted = !goal_first_k_explicit
                && goal_first_k_raw == 0
                && (candidates_mode == "lean-try"
                    || arg_value(rest, "--research-preset").is_some());
            let mut goal_first_k = if goal_first_k_defaulted {
                3
            } else {
                goal_first_k_raw
            };
            let fill_mode_raw = arg_value(rest, "--fill-mode");
            let focus_line_override = arg_u64(rest, "--focus-line").map(|x| x as usize);
            let focus_decl_override = arg_value(rest, "--focus-decl");
            let focus_decl_strict = arg_flag(rest, "--focus-decl-strict");
            let focus_decl_hard = arg_flag(rest, "--focus-decl-hard");
            let max_candidates_per_node =
                arg_u64(rest, "--max-candidates-per-node").map(|x| x as usize);
            let verify_k = arg_u64(rest, "--verify-k").map(|x| x as usize);
            // Optional heuristic knobs (default off; meant for experimentation).
            // - goal_meta_penalty: penalize selecting holes whose target contains metavariables (?m / ?_)
            // - depth_bonus: small nudge to prefer deeper nodes when sorting frontier (best-first tie-break)
            let goal_meta_penalty = arg_u64(rest, "--goal-meta-penalty").unwrap_or(0) as i64;
            let depth_bonus = arg_u64(rest, "--depth-bonus").unwrap_or(0) as i64;
            let no_cache = arg_flag(rest, "--no-cache");
            let cache_dir_opt = arg_value(rest, "--cache-dir").map(PathBuf::from);
            let profile = arg_flag(rest, "--profile");
            let summary_level = arg_u64(rest, "--summary-level").unwrap_or(2);
            let report_md_requested = arg_value(rest, "--report-md").map(PathBuf::from);
            let mut report_md = report_md_requested.clone();
            let llm_summary = arg_flag(rest, "--llm-summary");
            let llm_summary_timeout_s = arg_u64(rest, "--llm-summary-timeout-s").unwrap_or(20);
            let llm_planner = arg_flag(rest, "--llm-planner");
            // Only used when compiled with `--features planner`.
            #[cfg(feature = "planner")]
            let llm_planner_timeout_s = arg_u64(rest, "--llm-planner-timeout-s").unwrap_or(10);
            #[cfg(not(feature = "planner"))]
            let _llm_planner_timeout_s = arg_u64(rest, "--llm-planner-timeout-s").unwrap_or(10);
            let smt_precheck_on = arg_flag(rest, "--smt-precheck");
            let smt_precheck_off = arg_flag(rest, "--no-smt-precheck");
            let smt_solver_explicit = arg_value(rest, "--smt-solver").is_some();
            let mut smt_solver =
                arg_value(rest, "--smt-solver").unwrap_or_else(|| "auto".to_string());
            let smt_aggressive = arg_flag(rest, "--smt-aggressive");
            let mut smt_unsat_core = arg_flag(rest, "--smt-unsat-core");
            let mut smt_unsat_core_source = if smt_unsat_core { "explicit" } else { "off" };
            let mut smt_unsat_core_max = arg_u64(rest, "--smt-unsat-core-max")
                .unwrap_or(12)
                .clamp(0, 64) as usize;
            let mut smt_unsat_core_max_source = if arg_value(rest, "--smt-unsat-core-max").is_some()
            {
                "explicit"
            } else {
                "default"
            };
            let smt_support_explicit = arg_flag(rest, "--smt-support");
            let mut smt_support = smt_support_explicit;
            let mut smt_support_source = if smt_support { "explicit" } else { "off" };
            let smt_support_max =
                arg_u64(rest, "--smt-support-max").unwrap_or(8).clamp(0, 64) as usize;
            let smt_proof_explicit = arg_flag(rest, "--smt-proof");
            let mut smt_proof = smt_proof_explicit;
            let mut smt_proof_source = if smt_proof { "explicit" } else { "off" };
            let smt_proof_max_chars = arg_u64(rest, "--smt-proof-max-chars")
                .unwrap_or(12_000)
                .clamp(0, 200_000) as usize;
            let smt_proof_dump = arg_flag(rest, "--smt-proof-dump");
            let smt_proof_dump_dir_opt = arg_value(rest, "--smt-proof-dump-dir").map(PathBuf::from);
            let smt_proof_dump_max_chars = arg_u64(rest, "--smt-proof-dump-max-chars")
                .unwrap_or(200_000)
                .clamp(0, 5_000_000) as usize;
            if smt_proof_dump && !smt_proof {
                smt_proof = true;
                smt_proof_source = "implied_by_dump_flag";
            }
            if arg_value(rest, "--smt-repro-dir").is_some() && !smt_proof {
                smt_proof = true;
                smt_proof_source = "smt_repro_default";
            }
            let mut smt_dump = arg_flag(rest, "--smt-dump");
            let mut smt_dump_source = if smt_dump { "explicit" } else { "off" };
            let mut smt_dump_max =
                arg_u64(rest, "--smt-dump-max").unwrap_or(4).clamp(0, 64) as usize;
            let mut smt_dump_max_source = if arg_value(rest, "--smt-dump-max").is_some() {
                "explicit"
            } else {
                "default"
            };
            let smt_dump_dir_opt = arg_value(rest, "--smt-dump-dir").map(PathBuf::from);
            // If the user opted into unsat cores, default to producing at least one reproducible
            // SMT-LIB2 artifact (bounded).
            if smt_unsat_core && !smt_dump {
                smt_dump = true;
                smt_dump_source = "unsat_core_default";
                if !arg_value(rest, "--smt-dump-max").is_some() {
                    smt_dump_max = 1;
                    smt_dump_max_source = "unsat_core_default";
                }
            }
            // Aggressive mode: spend more resources to make SMT “strong”.
            if smt_aggressive {
                if !smt_unsat_core {
                    smt_unsat_core = true;
                    smt_unsat_core_source = "aggressive_default";
                }
                if !smt_proof {
                    smt_proof = true;
                    smt_proof_source = "aggressive_default";
                }
                if !arg_value(rest, "--smt-unsat-core-max").is_some() {
                    smt_unsat_core_max = smt_unsat_core_max.max(16);
                    smt_unsat_core_max_source = "aggressive_default";
                }
                if !smt_dump {
                    smt_dump = true;
                    smt_dump_source = "aggressive_default";
                }
                if !arg_value(rest, "--smt-dump-max").is_some() {
                    smt_dump_max = smt_dump_max.max(8);
                    smt_dump_max_source = "aggressive_default";
                }
            }
            let mut smt_solver_source = if smt_solver_explicit {
                "explicit"
            } else if arg_value(rest, "--research-preset").is_some() {
                "research_preset_default"
            } else if candidates_mode == "lean-try" {
                "lean_try_default"
            } else {
                "default"
            };
            let smt_timeout_ms_explicit = arg_value(rest, "--smt-timeout-ms").is_some();
            let mut smt_timeout_ms = arg_u64(rest, "--smt-timeout-ms").unwrap_or(1500);
            let mut smt_timeout_ms_source = if smt_timeout_ms_explicit {
                "explicit"
            } else if arg_value(rest, "--research-preset").is_some() {
                "research_preset_default"
            } else if candidates_mode == "lean-try" {
                "lean_try_default"
            } else {
                "default"
            };
            let smt_seed_explicit = arg_value(rest, "--smt-seed").is_some();
            let smt_seed = arg_u64(rest, "--smt-seed").unwrap_or(0);
            let smt_depth_explicit = arg_value(rest, "--smt-depth").is_some();
            let smt_depth_raw = arg_u64(rest, "--smt-depth").unwrap_or(0) as usize;
            let llm_timeout_s = arg_u64(rest, "--llm-timeout-s").unwrap_or(60);
            let goal_dump_raw = arg_flag(rest, "--goal-dump");
            let escalate_llm_flag = arg_flag(rest, "--escalate-llm");
            let escalate_llm_source = if escalate_llm_flag {
                "explicit"
            } else if arg_value(rest, "--research-preset").is_some() {
                "research_preset_default"
            } else {
                "off"
            };
            // (Compute smt/goal_dump sources after resolving research_preset + candidates_mode.)
            let escalate_llm = escalate_llm_flag;
            let allow_sorry_candidates = arg_flag(rest, "--allow-sorry-candidates");
            let include_trace = arg_flag(rest, "--include-trace");
            let pick = arg_value(rest, "--pick").unwrap_or_else(|| "best".to_string());
            let quiet = arg_flag(rest, "--quiet");
            let research_notes_file = arg_value(rest, "--research-notes-file").map(PathBuf::from);
            let research_preset = arg_value(rest, "--research-preset");
            let research_top_k = arg_u64(rest, "--research-top-k").unwrap_or(3) as usize;
            let include_diff = arg_flag(rest, "--include-diff");
            let diff_context = arg_u64(rest, "--diff-context").unwrap_or(3) as usize;
            let output_diff_requested = arg_value(rest, "--output-diff").map(PathBuf::from);
            let mut output_diff = output_diff_requested.clone();
            let write = arg_flag(rest, "--write");
            let write_to = arg_value(rest, "--write-to").map(PathBuf::from);
            let include_raw_verify = arg_flag(rest, "--include-raw-verify");
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);
            let smt_repro_dir_opt = arg_value(rest, "--smt-repro-dir").map(PathBuf::from);
            let smt_repro_dir_source = if smt_repro_dir_opt.is_some() {
                "explicit"
            } else {
                "off"
            };

            // SMT precheck defaults:
            // - baseline: on (best-effort, bounded; no solver means no effect)
            // - research preset: on
            // - lean-try mode: on (user opted into Lean-backed search; SMT usually pays here too)
            let smt_precheck = if smt_precheck_off {
                false
            } else {
                true || smt_precheck_on
                    || research_preset.is_some()
                    || candidates_mode == "lean-try"
            };
            let smt_precheck_source = if smt_precheck_off {
                "explicit_off"
            } else if smt_precheck_on {
                "explicit_on"
            } else if research_preset.is_some() {
                "research_preset_default"
            } else if candidates_mode == "lean-try" {
                "lean_try_default"
            } else {
                "baseline_default"
            };
            // If SMT is on by baseline default, keep timeout small unless explicitly overridden.
            if !smt_timeout_ms_explicit
                && research_preset.is_none()
                && candidates_mode != "lean-try"
                && smt_precheck_source == "baseline_default"
            {
                smt_timeout_ms = 250;
                smt_timeout_ms_source = "baseline_default";
            }
            if smt_aggressive && !smt_timeout_ms_explicit {
                smt_timeout_ms = smt_timeout_ms.max(8000);
                smt_timeout_ms_source = "aggressive_default";
            }

            // SMT support defaults:
            // - baseline: off (expensive)
            // - research preset: on (turn entailment into actionable Lean guidance)
            // - lean-try mode: on (user opted into heavier automation; support sets usually pay)
            if !smt_support_explicit {
                if research_preset.is_some() {
                    smt_support = true;
                    smt_support_source = "research_preset_default";
                } else if candidates_mode == "lean-try" {
                    smt_support = true;
                    smt_support_source = "lean_try_default";
                } else if smt_aggressive {
                    smt_support = true;
                    smt_support_source = "aggressive_default";
                }
            }

            // Goal dump defaults:
            // - baseline: off
            // - research preset: on
            // - lean-try mode: on (we already spend Lean budget; goal snapshots help ranking + debugging)
            let goal_dump_defaulted_by_repro = smt_repro_dir_opt.is_some()
                && !goal_dump_raw
                && research_preset.is_none()
                && candidates_mode != "lean-try";
            let goal_dump = goal_dump_raw
                || goal_dump_defaulted_by_repro
                || research_preset.is_some()
                || candidates_mode == "lean-try";
            let goal_dump_source = if goal_dump_raw {
                "explicit"
            } else if goal_dump_defaulted_by_repro {
                "smt_repro_default"
            } else if research_preset.is_some() {
                "research_preset_default"
            } else if candidates_mode == "lean-try" {
                "lean_try_default"
            } else {
                "off"
            };

            // SMT depth defaults:
            // - baseline: 0 (use all parseable LIA hyps)
            // - research preset: 2 (bounded var-connectivity closure; faster + less irrelevant noise)
            let mut smt_depth_source = if smt_depth_explicit {
                "explicit"
            } else if research_preset.is_some() {
                "research_preset_default"
            } else if candidates_mode == "lean-try" {
                "lean_try_default"
            } else {
                "default"
            };
            let smt_depth_defaulted = !smt_depth_explicit
                && smt_depth_raw == 0
                && (research_preset.is_some() || candidates_mode == "lean-try");
            let mut smt_depth = if smt_depth_defaulted {
                2
            } else {
                smt_depth_raw
            };
            if smt_aggressive && !smt_depth_explicit {
                smt_depth = smt_depth.max(2);
                smt_depth_source = "aggressive_default";
            }

            // Apply research preset policy overrides (if present).
            // (These only take effect when the corresponding CLI flag was not explicitly set.)
            if let Some(preset_name) = research_preset.as_ref() {
                if let Ok(Some(cfg)) = plc::config::load_from_repo_root(&repo_root) {
                    if let Some(preset) = cfg.research.resolve_preset(preset_name) {
                        if let Some(ts) = preset.tree_search {
                            if !smt_solver_explicit {
                                if let Some(v) = ts.smt_solver {
                                    smt_solver = v;
                                    smt_solver_source = "research_config_override";
                                }
                            }
                        }
                    }
                }
            }

            // Configure `smtkit` solver choice for this CLI invocation.
            // This must happen before any SMT probe/session is created.
            let smt_solver_norm = smt_solver.trim().to_lowercase();
            if smt_solver_norm != "auto" && !smt_solver_norm.is_empty() {
                let cmdline = match smt_solver_norm.as_str() {
                    "z3" => "z3 -in -smt2".to_string(),
                    "cvc5" => "cvc5 --lang smt2 --incremental".to_string(),
                    _ => smt_solver.trim().to_string(),
                };
                std::env::set_var("SMTKIT_SOLVER", cmdline);
            }

            // SMT explanation defaults:
            // - baseline: off (keep hot path fast)
            // - research preset: on (we want SMT to be auditable when it drives choices)
            let smt_explain_explicit =
                arg_flag(rest, "--smt-explain") || arg_flag(rest, "--no-smt-explain");
            let mut smt_explain_source = if arg_flag(rest, "--no-smt-explain") {
                "explicit_off"
            } else if arg_flag(rest, "--smt-explain") {
                "explicit"
            } else if research_preset.is_some() {
                "research_preset_default"
            } else if candidates_mode == "lean-try" && smt_depth_defaulted {
                "lean_try_default"
            } else {
                "default"
            };
            let mut smt_explain = if arg_flag(rest, "--no-smt-explain") {
                false
            } else {
                arg_flag(rest, "--smt-explain")
                    || research_preset.is_some()
                    || (candidates_mode == "lean-try" && smt_depth_defaulted)
            };
            let smt_explain_max_hyps_explicit = arg_value(rest, "--smt-explain-max-hyps").is_some();
            let mut smt_explain_max_hyps = arg_u64(rest, "--smt-explain-max-hyps")
                .unwrap_or(12)
                .clamp(0, 64) as usize;
            let mut smt_explain_max_hyps_source = if smt_explain_max_hyps_explicit {
                "explicit"
            } else if arg_value(rest, "--research-preset").is_some() {
                "research_preset_default"
            } else if candidates_mode == "lean-try" {
                "lean_try_default"
            } else {
                "default"
            };

            if write && write_to.is_some() {
                return Err("use only one of --write or --write-to".to_string());
            }
            if focus_line_override.is_some() && focus_decl_override.is_some() {
                return Err("use only one of --focus-line or --focus-decl".to_string());
            }
            if focus_decl_strict && focus_decl_override.is_none() {
                return Err("--focus-decl-strict requires --focus-decl".to_string());
            }
            if focus_decl_hard && focus_decl_override.is_none() {
                return Err("--focus-decl-hard requires --focus-decl".to_string());
            }

            if beam == 0 {
                return Err("--beam must be >= 1".to_string());
            }
            if max_nodes == 0 {
                return Err("--max-nodes must be >= 1".to_string());
            }
            if depth == 0 {
                return Err("--depth must be >= 1".to_string());
            }

            let repo_root =
                plc::find_lean_repo_root(&repo_root).map_err(|e| format!("repo_root: {e}"))?;
            plc::load_dotenv_smart(&repo_root);

            // Best-effort: create the dump directory up-front so `--smt-proof-dump-dir` is
            // guaranteed to exist even if we end up capturing 0 proofs.
            if smt_proof_dump {
                let root = smt_proof_dump_root(&repo_root, &smt_proof_dump_dir_opt);
                let _ = std::fs::create_dir_all(&root);
            }

            let cache_dir = if no_cache {
                None
            } else if let Some(p) = cache_dir_opt {
                Some(if p.is_absolute() {
                    p
                } else {
                    repo_root.join(p)
                })
            } else {
                Some(repo_root.join(".generated").join("proofpatch-cache"))
            };

            let abs = repo_root.join(&file);
            if !abs.exists() {
                return Err(format!("File not found: {}", abs.display()));
            }
            let original_text = std::fs::read_to_string(&abs)
                .map_err(|e| format!("read {}: {e}", abs.display()))?;

            // If the file has no `sorry`/`admit`, `tree-search-nearest` is not applicable.
            // Return a small structured result instead of erroring (useful for “repo is sorry-free” workflows).
            let locs_any = plc::locate_sorries_in_text(&original_text, 200, 1)?;
            let primary_any = plc::select_primary_sorry(focus_line_override, &locs_any);
            if primary_any.is_none() {
                let out = json!({
                    "ok": true,
                    "kind": "tree_search_nearest",
                    "result_kind": "early_no_sorries",
                    "repo_root": repo_root.display().to_string(),
                    "file": file,
                    "note": "No `sorry`/`admit` tokens found in file; nothing to patch.",
                    "sorries": 0,
                    "errors": 0,
                    "focus": {
                        "requested_decl": focus_decl_override,
                        "strict": focus_decl_strict,
                        "hard": focus_decl_hard,
                        "note": "focus is not applicable (file has no sorries)",
                    }
                });
                if let Some(p) = output_json {
                    write_json(&p, &out)?;
                    println!(
                        "{}",
                        json!({
                            "ok": true,
                            "written": p.display().to_string(),
                            "kind": out["kind"],
                            "result_kind": out["result_kind"],
                        })
                        .to_string()
                    );
                } else {
                    println!("{}", out.to_string());
                }
                return Ok(());
            }

            // Ensure the preflight goal dump targets the same hole we intend to search.
            //
            // If the user used `--focus-decl`, we may still want a goal dump at that decl’s nearest
            // `sorry` (even though `focus_line_override` is unset). This matters because:
            // - goal dumps populate cache-only signals (SMT precheck, goal sig stabilization)
            // - without this, we can end up dumping a different `sorry` and then SMT/ranking
            //   becomes unavailable for the actual focused hole.
            let mut focus_line_for_goal_dump = focus_line_override;
            if focus_line_for_goal_dump.is_none() {
                if let Some(fd0) = focus_decl_override.as_ref() {
                    let fd = fd0.trim().to_string();
                    let fd_last = fd
                        .split(|c| c == '.' || c == ':')
                        .filter(|s| !s.is_empty())
                        .last()
                        .unwrap_or(fd.as_str())
                        .to_string();
                    let needle = format!(".{fd_last}");
                    let picked = locs_any.iter().find(|l| {
                        l.decl_name
                            .as_ref()
                            .is_some_and(|dn| dn == &fd || dn == &fd_last || dn.ends_with(&needle))
                    });
                    if let Some(p) = picked {
                        focus_line_for_goal_dump = Some(p.line);
                    } else {
                        // Fallback: dump at the file’s primary sorry.
                        focus_line_for_goal_dump = primary_any.as_ref().map(|p| p.line);
                    }
                }
            }

            // Strict focus should fail fast *before* any expensive verification.
            // (Otherwise, users pay a baseline verify just to discover a typo in the decl name.)
            if focus_decl_strict {
                let fd = focus_decl_override
                    .clone()
                    .ok_or_else(|| "--focus-decl-strict requires --focus-decl".to_string())?;
                let fd = fd.trim().to_string();
                let fd_last = fd
                    .split(|c| c == '.' || c == ':')
                    .filter(|s| !s.is_empty())
                    .last()
                    .unwrap_or(fd.as_str())
                    .to_string();
                let needle = format!(".{fd_last}");
                let found = locs_any.iter().any(|l| {
                    l.decl_name
                        .as_ref()
                        .is_some_and(|dn| dn == &fd || dn == &fd_last || dn.ends_with(&needle))
                });
                if !found {
                    // Distinguish “typo” vs “decl exists but has no `sorry`”.
                    let decl_exists_in_file = original_text.lines().any(|ln| {
                        let t = ln.trim_start();
                        let mut it = t.split_whitespace();
                        let kw = it.next().unwrap_or("");
                        if !matches!(kw, "theorem" | "lemma" | "def") {
                            return false;
                        }
                        let name = it.next().unwrap_or("");
                        name == fd || name == fd_last || name.ends_with(&needle)
                    });

                    let mut available: Vec<String> = locs_any
                        .iter()
                        .filter_map(|l| l.decl_name.clone())
                        .collect::<std::collections::HashSet<_>>()
                        .into_iter()
                        .collect();
                    available.sort();
                    let available_short: Vec<String> = available.into_iter().take(24).collect();
                    let out = json!({
                        "ok": false,
                        "kind": "tree_search_nearest",
                        "result_kind": if decl_exists_in_file { "early_focus_decl_strict_empty" } else { "early_focus_decl_strict_not_found" },
                        "repo_root": repo_root.display().to_string(),
                        "file": file,
                        "error": if decl_exists_in_file {
                            "focus_decl_strict: decl exists but contains no `sorry`"
                        } else {
                            "focus_decl_strict: requested decl not found among sorry locations"
                        },
                        "focus": {
                            "source": if decl_exists_in_file { "focus_decl_strict_empty" } else { "focus_decl_strict_not_found" },
                            "requested_decl": fd,
                            "available_decls": available_short,
                        }
                    });
                    if let Some(p) = output_json.clone() {
                        write_json(&p, &out)?;
                        println!(
                            "{}",
                            json!({
                                "ok": false,
                                "written": p.display().to_string(),
                                "kind": out["kind"],
                                "result_kind": out["result_kind"],
                            })
                            .to_string()
                        );
                    } else {
                        println!("{}", out.to_string());
                    }
                    return Err(if decl_exists_in_file {
                        "focus_decl_strict: decl exists but empty".to_string()
                    } else {
                        "focus_decl_strict: decl not found".to_string()
                    });
                }
            }

            // Hard focus should also decide “do not drift” before expensive verification.
            // Unlike strict focus, this is a *successful* early stop (ok=true) with a clear note.
            if focus_decl_hard {
                let fd = focus_decl_override
                    .clone()
                    .ok_or_else(|| "--focus-decl-hard requires --focus-decl".to_string())?;
                let fd = fd.trim().to_string();
                let fd_last = fd
                    .split(|c| c == '.' || c == ':')
                    .filter(|s| !s.is_empty())
                    .last()
                    .unwrap_or(fd.as_str())
                    .to_string();
                let needle = format!(".{fd_last}");
                let found = locs_any.iter().any(|l| {
                    l.decl_name
                        .as_ref()
                        .is_some_and(|dn| dn == &fd || dn == &fd_last || dn.ends_with(&needle))
                });
                if !found {
                    // Distinguish:
                    // - decl not found anywhere in the file (likely typo)
                    // - decl exists, but it contains no `sorry` (nothing to patch in that decl)
                    let decl_exists_in_file = original_text.lines().any(|ln| {
                        let t = ln.trim_start();
                        let mut it = t.split_whitespace();
                        let kw = it.next().unwrap_or("");
                        if !matches!(kw, "theorem" | "lemma" | "def") {
                            return false;
                        }
                        let name = it.next().unwrap_or("");
                        name == fd || name == fd_last || name.ends_with(&needle)
                    });

                    let mut available: Vec<String> = locs_any
                        .iter()
                        .filter_map(|l| l.decl_name.clone())
                        .collect::<std::collections::HashSet<_>>()
                        .into_iter()
                        .collect();
                    available.sort();
                    let available_short: Vec<String> = available.into_iter().take(24).collect();
                    let out = json!({
                        "ok": true,
                        "kind": "tree_search_nearest",
                        "result_kind": if decl_exists_in_file { "early_focus_decl_hard_empty" } else { "early_focus_decl_hard_not_found" },
                        "repo_root": repo_root.display().to_string(),
                        "file": file,
                        "note": if decl_exists_in_file {
                            "Hard focus: requested decl exists but contains no `sorry`; nothing to patch (refusing to drift)."
                        } else {
                            "Hard focus: requested decl did not match any `sorry` location; refusing to drift to other declarations."
                        },
                        "sorries": locs_any.len(),
                        "focus": {
                            "source": if decl_exists_in_file { "focus_decl_hard_empty" } else { "focus_decl_hard_not_found" },
                            "requested_decl": fd,
                            "available_decls": available_short,
                            "hard": true
                        }
                    });
                    if let Some(p) = output_json.clone() {
                        write_json(&p, &out)?;
                        println!(
                            "{}",
                            json!({
                                "ok": true,
                                "written": p.display().to_string(),
                                "kind": out["kind"],
                                "result_kind": out["result_kind"],
                            })
                            .to_string()
                        );
                    } else {
                        println!("{}", out.to_string());
                    }
                    return Ok(());
                }
            }

            // Local default: when caching is enabled, auto-write useful artifacts unless disabled.
            // This makes `.generated/*.json` “actionable” even when the caller didn't pass paths.
            let auto_artifacts = arg_flag(rest, "--auto-artifacts")
                || env_truthy("PROOFPATCH_TREE_SEARCH_AUTO_ARTIFACTS", true);
            let run_key = hash_text(&format!(
                "{}|focus_line={:?}|beam={}|max_nodes={}|depth={}|candidates={}",
                file, focus_line_override, beam, max_nodes, depth, candidates_mode
            ));
            let mk_path = |subdir: &str, ext: &str| -> Option<PathBuf> {
                cache_dir.as_ref().map(|root| {
                    root.join("tree_search")
                        .join(subdir)
                        .join(format!("{run_key}.{ext}"))
                })
            };
            if auto_artifacts {
                if events_jsonl.is_none() {
                    events_jsonl = mk_path("events", "jsonl");
                }
                if report_md.is_none() {
                    report_md = mk_path("reports", "md");
                }
                if include_diff && output_diff.is_none() {
                    output_diff = mk_path("diffs", "diff");
                }
            }

            let mut research_notes: Option<serde_json::Value> = None;
            let mut research_notes_text: Option<String> = None;
            if let Some(p) = research_notes_file.as_ref() {
                let s = std::fs::read_to_string(p)
                    .map_err(|e| format!("read research notes {}: {e}", p.display()))?;
                let max_chars = 12_000usize;
                let (kept, truncated) = if s.chars().count() > max_chars {
                    let kept: String = s.chars().take(max_chars).collect();
                    (kept, true)
                } else {
                    (s, false)
                };
                let mut txt = kept;
                if truncated {
                    txt.push_str("\n\n[proofpatch: research_notes truncated]\n");
                }
                research_notes_text = Some(txt);
                research_notes = Some(json!({
                    "path": p.display().to_string(),
                    "truncated": truncated,
                    "chars": research_notes_text.as_ref().map(|s| s.chars().count()).unwrap_or(0),
                }));
            }

            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| format!("failed to build tokio runtime: {e}"))?;

            // Repo-owned research preset → bounded prompt context (and traceable metadata in output).
            //
            // This makes “LL ops” actionable: research is not just hydrated; it directly conditions
            // the LLM calls inside this loop (planner/escalation), and we record the context used.
            let mut research_context: Option<serde_json::Value> = None;
            if let Some(preset_name) = research_preset.as_deref() {
                if let Some(cfg) = plc::config::load_from_repo_root(&repo_root)? {
                    if let Some(preset) = cfg.research.resolve_preset(preset_name) {
                        // Merge preset-provided tree-search policy with our good defaults:
                        // - explicit CLI flags win
                        // - then preset policy (merged with research.defaults.tree_search)
                        // - then mode defaults (lean-try / research mode)
                        if let Some(ts) = preset.tree_search.clone() {
                            if !goal_first_k_explicit {
                                if let Some(v) = ts.goal_first_k {
                                    goal_first_k = v;
                                    goal_first_k_source = "research_config_override";
                                }
                            }
                            if !smt_depth_explicit {
                                if let Some(v) = ts.smt_depth {
                                    smt_depth = v;
                                    smt_depth_source = "research_config_override";
                                }
                            }
                            if !smt_timeout_ms_explicit {
                                if let Some(v) = ts.smt_timeout_ms {
                                    smt_timeout_ms = v;
                                    smt_timeout_ms_source = "research_config_override";
                                }
                            }
                            if !smt_seed_explicit {
                                // (rare) allow a preset to force determinism for SMT heuristics
                                // by pinning the seed.
                                // NOTE: we only override when user didn't set --smt-seed.
                                // (No field today; reserved for future.)
                            }
                            if !smt_explain_explicit {
                                if let Some(v) = ts.smt_explain {
                                    smt_explain = v;
                                    smt_explain_source = "research_config_override";
                                }
                            }
                            if !smt_explain_max_hyps_explicit {
                                if let Some(v) = ts.smt_explain_max_hyps {
                                    smt_explain_max_hyps = v.min(64);
                                    smt_explain_max_hyps_source = "research_config_override";
                                }
                            }
                        }
                        let mut papers = rt.block_on(plc::arxiv::arxiv_search(
                            &preset.query,
                            preset.max_results,
                            StdDuration::from_millis(preset.timeout_ms),
                        ))?;

                        let must_any_l: Vec<String> = preset
                            .must_include_any
                            .iter()
                            .map(|s| s.to_lowercase())
                            .collect();
                        let must_all_l: Vec<String> = preset
                            .must_include_all
                            .iter()
                            .map(|s| s.to_lowercase())
                            .collect();
                        if !must_any_l.is_empty() || !must_all_l.is_empty() {
                            papers = papers
                                .into_iter()
                                .filter(|p| {
                                    let hay =
                                        format!("{}\n{}", p.title, p.abstract_text).to_lowercase();
                                    let ok_any = must_any_l.is_empty()
                                        || must_any_l.iter().any(|tok| hay.contains(tok));
                                    let ok_all = must_all_l.is_empty()
                                        || must_all_l.iter().all(|tok| hay.contains(tok));
                                    ok_any && ok_all
                                })
                                .collect();
                        }

                        let mut ctx = json!({
                            "ok": true,
                            "kind": "research_context",
                            "preset": preset_name,
                            "settings": preset,
                            "arxiv": {
                                "query": preset.query,
                                "max_results": preset.max_results,
                                "timeout_ms": preset.timeout_ms,
                                "filter": { "must_include_any": preset.must_include_any, "must_include_all": preset.must_include_all },
                                "papers": papers,
                            }
                        });

                        if preset.llm_summary {
                            let kind = preset
                                .llm_summary_kind
                                .as_deref()
                                .unwrap_or(research_summary_kind_default());
                            let system = research_summary_system_prompt(kind);
                            let user = serde_json::to_string(&json!({
                                "preset": preset_name,
                                "query": ctx["arxiv"]["query"],
                                "papers": ctx["arxiv"]["papers"],
                            }))
                            .unwrap_or_else(|_| "{\"papers\":[]}".to_string());
                            let res = match normalize_summary_kind(kind).as_str() {
                                "formalization_v2" => rt
                                    .block_on(plc::llm::chat_completion_structured::<
                                        ResearchSummaryV2,
                                    >(
                                        &system,
                                        &user,
                                        StdDuration::from_secs(preset.llm_timeout_s),
                                    ))
                                    .map(|r| {
                                        let capped = cap_summary_v2(r.value, &preset);
                                        let v = serde_json::to_value(&capped)
                                            .unwrap_or(serde_json::Value::Null);
                                        (
                                            r.provider,
                                            r.model,
                                            r.model_source,
                                            r.model_env,
                                            r.mode,
                                            v,
                                        )
                                    }),
                                _ => rt
                                    .block_on(
                                        plc::llm::chat_completion_structured::<ResearchSummary>(
                                            &system,
                                            &user,
                                            StdDuration::from_secs(preset.llm_timeout_s),
                                        ),
                                    )
                                    .map(|r| {
                                        let capped = cap_summary_v1(r.value, &preset);
                                        let v = serde_json::to_value(&capped)
                                            .unwrap_or(serde_json::Value::Null);
                                        (
                                            r.provider,
                                            r.model,
                                            r.model_source,
                                            r.model_env,
                                            r.mode,
                                            v,
                                        )
                                    }),
                            };
                            if let Ok((provider, model, model_source, model_env, mode, v)) = res {
                                ctx["llm_summary"] = json!({
                                    "provider": provider,
                                    "model": model,
                                    "model_source": model_source,
                                    "model_env": model_env,
                                    "mode": mode,
                                    "kind": kind,
                                    "content_struct": v,
                                });
                            }
                        }

                        let notes = plc::ingest_research_json(&ctx);
                        let notes_v =
                            serde_json::to_value(&notes).unwrap_or(serde_json::Value::Null);
                        ctx["notes"] = notes_v.clone();

                        // Merge into the prompt notes (bounded).
                        let mut s = String::new();
                        s.push_str("Research context (repo-owned preset):\n");
                        s.push_str(&format!("- preset: {preset_name}\n"));
                        s.push_str(&format!("- query: {}\n", preset.query));
                        s.push_str(&format!("- papers: {}\n", papers.len()));
                        if let Some(xs) = notes_v.get("sources").and_then(|v| v.as_array()) {
                            s.push_str("\nTop sources:\n");
                            for src in xs.iter().take(research_top_k) {
                                let title = src.get("title").and_then(|v| v.as_str()).unwrap_or("");
                                let url = src
                                    .get("canonical_url")
                                    .and_then(|v| v.as_str())
                                    .or_else(|| src.get("url").and_then(|v| v.as_str()))
                                    .unwrap_or("");
                                if !title.is_empty() || !url.is_empty() {
                                    s.push_str(&format!("- {} {}\n", title, url));
                                }
                            }
                        }
                        if let Some(llm) =
                            ctx.get("llm_summary").and_then(|v| v.get("content_struct"))
                        {
                            if let Some(pitfalls) = llm.get("pitfalls") {
                                s.push_str("\nPitfalls:\n");
                                s.push_str(&format!("{}\n", pitfalls));
                            }
                            if let Some(shape) = llm.get("proof_shape") {
                                s.push_str("\nProof shape:\n");
                                s.push_str(&format!("{}\n", shape));
                            }
                            if let Some(ids) = llm.get("mathlib_idents") {
                                s.push_str("\nMathlib idents:\n");
                                s.push_str(&format!("{}\n", ids));
                            }
                        }

                        research_notes_text = match research_notes_text {
                            Some(prev) => Some(format!("{s}\n\nExternal notes file:\n{prev}")),
                            None => Some(s),
                        };
                        research_notes = Some(json!({
                            "preset": preset_name,
                            "top_k": research_top_k,
                            "deduped_urls": notes.deduped_urls,
                        }));
                        research_context = Some(ctx);
                    }
                }
            }

            // Repo-configured hint packs (repo-specific accelerants, opt-in).
            // For tree-search, a research preset may override which packs are enabled.
            let hint_rules: Vec<plc::config::HintRule> =
                load_hint_rules(&repo_root, research_preset.as_deref());

            // Research preset implies “LLM is allowed to help if stuck” (bounded + best-effort).
            let escalate_llm = escalate_llm || research_preset.is_some();
            let decision_effects = json!({
                "candidates_mode": { "value": candidates_mode, "source": candidates_mode_source },
                "goal_first_k": { "value": goal_first_k, "source": goal_first_k_source },
                "smt_precheck": { "value": smt_precheck, "source": smt_precheck_source },
                "smt_aggressive": { "value": smt_aggressive, "source": if smt_aggressive { "explicit" } else { "off" } },
                "smt_solver": { "value": smt_solver, "source": smt_solver_source },
                "smt_unsat_core": { "value": smt_unsat_core, "source": smt_unsat_core_source },
                "smt_unsat_core_max": { "value": smt_unsat_core_max, "source": smt_unsat_core_max_source },
                "smt_support": { "value": smt_support, "source": smt_support_source },
                "smt_support_max": { "value": smt_support_max, "source": if arg_value(rest, "--smt-support-max").is_some() { "explicit" } else { "default" } },
                "smt_dump": { "value": smt_dump, "source": smt_dump_source },
                "smt_dump_max": { "value": smt_dump_max, "source": smt_dump_max_source },
                "smt_dump_dir": { "value": smt_dump_dir_opt.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "".to_string()), "source": if arg_value(rest, "--smt-dump-dir").is_some() { "explicit" } else { "default" } },
                "smt_timeout_ms": { "value": smt_timeout_ms, "source": smt_timeout_ms_source },
                "smt_depth": {
                    "value": smt_depth,
                    "source": smt_depth_source
                },
                "smt_explain": {
                    "value": smt_explain,
                    "source": smt_explain_source
                },
                "smt_explain_max_hyps": { "value": smt_explain_max_hyps, "source": smt_explain_max_hyps_source },
                "goal_dump": { "value": goal_dump, "source": goal_dump_source },
                "smt_repro_dir": {
                    "value": smt_repro_dir_opt
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "".to_string()),
                    "source": smt_repro_dir_source
                },
                "escalate_llm": { "value": escalate_llm, "source": escalate_llm_source },
            });

            // Optional lightweight profiling (accumulates ms). Controlled by `--profile`.
            let mut prof_verify_baseline_ms: u64 = 0;
            let mut prof_verify_nodes_ms: u64 = 0;
            let mut prof_goal_dump_ms: u64 = 0;
            let mut prof_lean_suggest_ms: u64 = 0;
            let mut prof_locate_sorries_ms: u64 = 0;
            let mut prof_conservative_sorries_ms: u64 = 0;
            let mut prof_patch_ms: u64 = 0;
            let mut prof_verify_baseline_calls: u64 = 0;
            let mut prof_verify_nodes_calls: u64 = 0;
            let mut prof_candidates_considered: u64 = 0;
            let mut prof_candidates_verified: u64 = 0;

            let mut events_tail: Vec<serde_json::Value> = Vec::new();
            let mut events_all: Vec<serde_json::Value> = Vec::new();
            let mut events_by_kind: std::collections::HashMap<String, u64> =
                std::collections::HashMap::new();
            let mut record_event = |kind: &str, mut v: serde_json::Value| {
                *events_by_kind.entry(kind.to_string()).or_insert(0) += 1;
                let t_ms = prof_t0.elapsed().as_millis() as u64;
                if let Some(obj) = v.as_object_mut() {
                    obj.entry("kind".to_string()).or_insert(json!(kind));
                    obj.entry("t_ms".to_string()).or_insert(json!(t_ms));
                }
                if events_tail.len() >= events_keep {
                    events_tail.remove(0);
                }
                events_tail.push(v);
                if events_all.len() < events_all_keep {
                    events_all.push(
                        events_tail
                            .last()
                            .cloned()
                            .unwrap_or(serde_json::Value::Null),
                    );
                }
            };

            record_event(
                "start",
                json!({
                    "repo_root": repo_root.display().to_string(),
                    "file": file,
                    "config": {
                        "timeout_s": timeout_s,
                        "total_timeout_s": total_timeout_s,
                        "goal_dump_timeout_s": goal_dump_timeout_s,
                        "oracle_timeout_s": oracle_timeout_s,
                        "beam": beam,
                        "max_nodes": max_nodes,
                        "depth": depth,
                        "candidates": candidates_mode,
                        "goal_first_k": goal_first_k,
                        "dedup_goal_expansions": dedup_goal_expansions,
                        "fill_mode": fill_mode_raw.as_deref().unwrap_or("auto"),
                        "log_level": log_level,
                        "research_preset": research_preset,
                        "research_top_k": research_top_k,
                        "escalate_llm": escalate_llm,
                    }
                }),
            );

            let mut goal_dump_v: Option<serde_json::Value> = None;
            let mut lean_suggest_v: Option<serde_json::Value> = None;
            if goal_dump || matches!(candidates_mode.trim(), "auto" | "lean") {
                if let Some(dur) = budget_dur(goal_dump_timeout_s) {
                    let gd = if let Some(fl) = focus_line_for_goal_dump {
                        rt.block_on(plc::goal_dump_in_text_at(
                            &repo_root,
                            &file,
                            &original_text,
                            dur,
                            Some(fl),
                            None,
                        ))
                        .map_err(|e| format!("goal_dump_in_text_at failed: {e}"))?
                    } else {
                        rt.block_on(plc::goal_dump_nearest(&repo_root, &file, dur))
                            .map_err(|e| format!("goal_dump_nearest failed: {e}"))?
                    };
                    goal_dump_v = Some(gd);
                } else {
                    bailed_total_timeout = true;
                    record_event(
                        "bailout_total_timeout",
                        json!({ "where": "goal_dump_preflight" }),
                    );
                }
            }

            // Candidate generator:
            // - det: fixed list
            // - auto: derive from pp_dump goal pretty (fallback to det)
            // - lean: use Lean/mathlib suggestions (`simp?`/`exact?`/`apply?`) (fallback to auto/det)
            // - llm: ask for JSON array (fallback to auto/det)
            let candidates_mode = candidates_mode.trim().to_lowercase();
            let lean_oracle_per_node = if candidates_mode == "lean-try" {
                // Advanced default for proof-tree search: use Lean oracle at each hole (bounded).
                true
            } else {
                lean_oracle_per_node
            };
            let goal_first_k =
                if candidates_mode == "lean-try" && goal_first_k == 0 && !goal_first_k_explicit {
                    // Default “goal-first” probing for LeanTree-ish behavior.
                    3
                } else {
                    goal_first_k
                };
            let fill_mode = if candidates_mode == "lean-try" {
                fill_mode_raw.unwrap_or_else(|| "hybrid".to_string())
            } else {
                fill_mode_raw.unwrap_or_else(|| "safe".to_string())
            };

            // Effective expansion budgets (make these explicit for grok + reproducibility).
            let effective_max_candidates_per_node = if candidates_mode == "lean-try" {
                max_candidates_per_node.unwrap_or(beam)
            } else {
                max_candidates_per_node.unwrap_or(usize::MAX)
            };
            let effective_verify_k = if candidates_mode == "lean-try" {
                verify_k.unwrap_or_else(|| (beam + 1) / 2)
            } else {
                verify_k.unwrap_or(usize::MAX)
            }
            .min(effective_max_candidates_per_node)
            .max(1);
            // Default to goal-state dedup for `lean-try`.
            let dedup_goal_expansions = if candidates_mode == "lean-try" {
                true
            } else {
                dedup_goal_expansions
            };
            // Default higher oracle budget for `lean-try` unless explicitly set.
            let lean_oracle_max_calls = if candidates_mode == "lean-try" {
                if arg_value(rest, "--lean-oracle-max-calls").is_some() {
                    lean_oracle_max_calls
                } else {
                    24
                }
            } else {
                lean_oracle_max_calls
            };

            record_event(
                "budgets",
                json!({
                    "candidates": candidates_mode,
                    "fill_mode": fill_mode,
                    "lean_oracle_per_node": lean_oracle_per_node,
                    "lean_oracle_max_calls": lean_oracle_max_calls,
                    "goal_first_k": goal_first_k,
                    "dedup_goal_expansions": dedup_goal_expansions,
                    "effective_max_candidates_per_node": if effective_max_candidates_per_node == usize::MAX { serde_json::Value::Null } else { json!(effective_max_candidates_per_node) },
                    "effective_verify_k": if effective_verify_k == usize::MAX { serde_json::Value::Null } else { json!(effective_verify_k) },
                }),
            );

            // Shared “safe tactic fallback” used for deeper proof-tree loops.
            // This is intentionally conservative: it tries a small set of tactics and then
            // falls back to an explicit `sorry` (honest intermediate state) so nodes stay `ok=true`.
            let safe_tactic_fallback_block = |indent: &str| -> String {
                format!(
                    "{indent}try (simp; done)\n{indent}try (norm_cast; done)\n{indent}try (aesop; done)\n{indent}try (omega; done)\n{indent}try (nlinarith; done)\n{indent}try (linarith; done)\n{indent}try (ring_nf; done)\n{indent}try (norm_num; done)\n{indent}sorry"
                )
            };

            // Single-line `first | ... | sorry` form for tactic holes (works well inside nested contexts).
            let safe_first_line = || -> String {
                "first | (simp; done) | (norm_cast; done) | (aesop; done) | (omega; done) | (nlinarith; done) | (linarith; done) | (ring_nf; done) | (norm_num; done) | sorry".to_string()
            };
            let mut llm_meta_initial: Option<serde_json::Value> = None;
            let mut llm_escalate_attempts: u64 = 0;
            let mut llm_escalate_successes: u64 = 0;
            let mut llm_escalate_last_error: Option<String> = None;
            let candidates = if candidates_mode == "auto" {
                if let Some(gd) = goal_dump_v.as_ref() {
                    let pretty = gd
                        .get("pp_dump")
                        .and_then(|v| v.get("goals"))
                        .and_then(|v| v.as_array())
                        .and_then(|xs| xs.first())
                        .and_then(|v| v.get("pretty"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let mut derived = plc::derive_candidates_from_goal_pretty_with_hint_rules(
                        pretty,
                        &hint_rules,
                    );
                    if derived.is_empty() {
                        derived = default_det_candidates();
                    }
                    sanitize_candidates(derived)
                } else {
                    sanitize_candidates(default_det_candidates())
                }
            } else if candidates_mode == "lean" {
                let ls = if let Some(dur) = budget_dur(oracle_timeout_s) {
                    rt.block_on(plc::lean_suggest_nearest(&repo_root, &file, dur))
                        .ok()
                } else {
                    bailed_total_timeout = true;
                    record_event(
                        "bailout_total_timeout",
                        json!({ "where": "lean_suggest_nearest" }),
                    );
                    None
                };
                lean_suggest_v = ls.clone();
                let mut xs = ls
                    .as_ref()
                    .and_then(|v| v.get("suggestions"))
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(|s| s.to_string()))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                xs = xs
                    .into_iter()
                    .map(|s| {
                        let t = s.trim();
                        if t.starts_with("by") {
                            t.to_string()
                        } else {
                            format!("by\n  {t}")
                        }
                    })
                    .collect();
                if xs.is_empty() {
                    if let Some(gd) = goal_dump_v.as_ref() {
                        let pretty = gd
                            .get("pp_dump")
                            .and_then(|v| v.get("goals"))
                            .and_then(|v| v.as_array())
                            .and_then(|xs| xs.first())
                            .and_then(|v| v.get("pretty"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        xs = plc::derive_candidates_from_goal_pretty_with_hint_rules(
                            pretty,
                            &hint_rules,
                        );
                    }
                }
                if xs.is_empty() {
                    xs = default_det_candidates();
                }
                sanitize_candidates(xs)
            } else if candidates_mode == "lean-try" {
                // `lean-try` is primarily driven by the *per-node* oracle (`lean_suggest_in_text_at`)
                // during the proof-tree loop. Seeding upfront suggestions can be surprisingly slow on
                // some repos, and can consume the entire `--total-timeout-s` before the search starts.
                //
                // Opt-in with:
                // - PROOFPATCH_LEAN_TRY_SEED_ORACLE=1
                let seed_oracle = std::env::var("PROOFPATCH_LEAN_TRY_SEED_ORACLE")
                    .ok()
                    .map(|s| {
                        matches!(
                            s.trim().to_lowercase().as_str(),
                            "1" | "true" | "yes" | "y" | "on"
                        )
                    })
                    .unwrap_or(false);

                let mut xs: Vec<String>;
                if seed_oracle {
                    record_event("oracle_seed_call", json!({ "timeout_s": oracle_timeout_s }));
                    let t0 = std::time::Instant::now();
                    let ls = if let Some(dur) = budget_dur(oracle_timeout_s) {
                        rt.block_on(plc::lean_suggest_nearest(&repo_root, &file, dur))
                            .ok()
                    } else {
                        bailed_total_timeout = true;
                        record_event(
                            "bailout_total_timeout",
                            json!({ "where": "lean_suggest_nearest_seed" }),
                        );
                        None
                    };
                    let ms = t0.elapsed().as_millis() as u64;
                    record_event(
                        "oracle_seed_result",
                        json!({
                            "ms": ms,
                            "ok": ls.is_some(),
                            "suggestions_n": ls.as_ref().and_then(|v| v.get("suggestions")).and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0),
                        }),
                    );
                    lean_suggest_v = ls.clone();
                    xs = ls
                        .as_ref()
                        .and_then(|v| v.get("suggestions"))
                        .and_then(|v| v.as_array())
                        .map(|a| {
                            a.iter()
                                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                } else {
                    record_event("oracle_seed_skipped", json!({ "reason": "default_off" }));
                    xs = default_det_candidates();
                }
                // Turn each oracle suggestion into a *structured skeleton*:
                // - keep the `refine ... ?_ ...` (or similar) line
                // - add one bullet per `?_` hole
                // - for each bullet, try a small deterministic tactic set, and fall back to `sorry`
                //
                // This enables recursive/beam refinement: the node is honest (explicit `sorry`s),
                // and further depth can patch the remaining subgoals.
                xs = xs
                    .into_iter()
                    .filter_map(|s| {
                        let t = s.trim();
                        let t = if t.starts_with("by") { t.to_string() } else { t.to_string() };
                        let holes = t.matches("?_").count().max(1).min(8);
                        let mut out = String::new();
                        out.push_str("by\n  ");
                        out.push_str(&t);
                        for _ in 0..holes {
                            out.push_str("\n  ·\n    try (simp; done)\n    try (aesop; done)\n    try (omega; done)\n    try (nlinarith; done)\n    try (linarith; done)\n    try (ring_nf; done)\n    try (norm_num; done)\n    sorry");
                        }
                        Some(out)
                    })
                    .collect();
                if xs.is_empty() {
                    xs = default_det_candidates();
                }
                sanitize_candidates(xs)
            } else if candidates_mode == "llm" {
                // Use the nearest-sorry region prompt to ask for multiple candidates.
                let locs0 = plc::locate_sorries_in_text(&original_text, 50, 1)?;
                let primary0 = plc::select_primary_sorry(None, &locs0)
                    .ok_or_else(|| "No `sorry`/`admit` tokens found in file.".to_string())?;
                let payload = plc::build_region_patch_prompt(
                    &repo_root,
                    &file,
                    primary0.region_start,
                    primary0.region_end,
                    None,
                )?;

                let mut system = payload.system.clone();
                system.push_str("\n\nReturn a JSON array of 6 distinct candidate Lean replacements (strings).\nEach element must be a proof term only (no markdown fences).");
                if !allow_sorry_candidates {
                    system.push_str("\n\nConstraints:\n- Do not use `sorry` or `admit` anywhere.\n- Return complete proof terms only (no placeholders).");
                }
                if let Some(gd) = goal_dump_v.as_ref() {
                    if let Some(pretty) = gd
                        .get("pp_dump")
                        .and_then(|v| v.get("goals"))
                        .and_then(|v| v.as_array())
                        .and_then(|xs| xs.first())
                        .and_then(|v| v.get("pretty"))
                        .and_then(|v| v.as_str())
                    {
                        system.push_str("\n\nGoal snapshot (pretty):\n");
                        system.push_str(pretty);
                    }
                }
                if let Some(notes) = research_notes_text.as_ref() {
                    system.push_str("\n\nResearch context (may be incomplete):\n");
                    system.push_str(notes);
                }

                let res = rt.block_on(plc::llm::chat_completion(
                    &system,
                    &payload.user,
                    StdDuration::from_secs(llm_timeout_s),
                ));

                let mut parsed: Option<Vec<String>> = None;
                match res {
                    Ok(done) => {
                        parsed = parse_json_string_array(&done.content);
                        // Count `sorry`/`admit` candidates so we can explain later filtering.
                        let (total, contains_sorry) = if let Some(xs) = parsed.as_ref() {
                            let mut bad = 0usize;
                            for x in xs {
                                let lc = x.to_lowercase();
                                if lc.contains("sorry") || lc.contains("admit") {
                                    bad += 1;
                                }
                            }
                            (xs.len(), bad)
                        } else {
                            (0usize, 0usize)
                        };
                        llm_meta_initial = Some(json!({
                            "attempted": true,
                            "ok": true,
                            "parsed": parsed.is_some(),
                            "error": if parsed.is_some() { serde_json::Value::Null } else { serde_json::Value::String("llm_response_not_json_string_array".to_string()) },
                            "response_preview": done.content.chars().take(400).collect::<String>(),
                            "parsed_counts": { "total": total, "contains_sorry_or_admit": contains_sorry }
                        }));
                    }
                    Err(e) => {
                        llm_meta_initial = Some(json!({
                            "attempted": true,
                            "ok": false,
                            "parsed": false,
                            "error": format!("{e}"),
                        }));
                    }
                }

                let xs = parsed
                    .or_else(|| {
                        // fallback: derived candidates if we have goal_dump
                        goal_dump_v.as_ref().and_then(|gd| {
                            let pretty = gd
                                .get("pp_dump")
                                .and_then(|v| v.get("goals"))
                                .and_then(|v| v.as_array())
                                .and_then(|xs| xs.first())
                                .and_then(|v| v.get("pretty"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            let derived = plc::derive_candidates_from_goal_pretty_with_hint_rules(
                                pretty,
                                &hint_rules,
                            );
                            if derived.is_empty() {
                                None
                            } else {
                                Some(derived)
                            }
                        })
                    })
                    .unwrap_or_else(default_det_candidates);
                sanitize_candidates(xs)
            } else {
                sanitize_candidates(default_det_candidates())
            };

            // Safety default: don't allow the search to “solve” holes with new holes.
            let mut candidates = candidates;
            // `lean-try` intentionally introduces explicit `sorry` as an honest intermediate state.
            let allow_sorry_in_candidates = allow_sorry_candidates || candidates_mode == "lean-try";
            if !allow_sorry_in_candidates {
                candidates.retain(|c| {
                    let lc = c.to_lowercase();
                    !lc.contains("sorry") && !lc.contains("admit")
                });
                // Never end up with an empty pool.
                if candidates.is_empty() {
                    candidates = sanitize_candidates(default_det_candidates());
                }
            }

            // `lean-try` is a two-phase strategy:
            // - depth 1: install a Lean-proposed skeleton (may include explicit `sorry`)
            // - deeper: fill subgoals with *safe* deterministic candidates (avoid re-installing skeletons)
            let (candidates_root, candidates_fill) = if candidates_mode == "lean-try" {
                let fm = fill_mode.trim().to_lowercase();
                // strict: actually try to solve (will error if it fails)
                let strict = vec![
                    "by\n  (simp; done)".to_string(),
                    "by\n  (simp_all; done)".to_string(),
                    "by\n  (norm_cast; done)".to_string(),
                    "by\n  (aesop; done)".to_string(),
                    "by\n  (omega; done)".to_string(),
                    "by\n  (nlinarith; done)".to_string(),
                    "by\n  (linarith; done)".to_string(),
                    "by\n  (ring_nf; done)".to_string(),
                    "by\n  (norm_num; done)".to_string(),
                    // try suggestion tactics (can succeed; we still gate success by sorry_warnings==0)
                    "by\n  (exact?; done)".to_string(),
                    "by\n  (apply?; done)".to_string(),
                    "by\n  classical\n  (exact?; done)".to_string(),
                    "by\n  classical\n  (apply?; done)".to_string(),
                ];
                // safe: never hard-error; keeps explicit `sorry` fallback (honest intermediate)
                let safe = vec![
                    "by\n  first | (simp; done) | (norm_cast; done) | (aesop; done) | (omega; done) | (nlinarith; done) | (linarith; done) | (ring_nf; done) | (norm_num; done) | sorry".to_string(),
                    "by\n  classical\n  first | (simp; done) | (norm_cast; done) | (aesop; done) | (omega; done) | (nlinarith; done) | (linarith; done) | (ring_nf; done) | (norm_num; done) | sorry".to_string(),
                ];
                let fill = match fm.as_str() {
                    "strict" => strict,
                    "hybrid" => {
                        let mut v = strict;
                        v.extend(safe);
                        v
                    }
                    _ => safe,
                };
                (candidates.clone(), sanitize_candidates(fill))
            } else {
                (candidates.clone(), candidates.clone())
            };

            // Per-node Lean oracle cache (bounded).
            // Keyed by (hash(text), len(text), focus_line).
            let mut lean_oracle_calls: usize = 0;
            let mut lean_oracle_cache_hits: usize = 0;
            let mut lean_oracle_cache_misses: usize = 0;
            let mut lean_oracle_goal_dedup_skips: usize = 0;
            let mut lean_oracle_cache: std::collections::HashMap<(u64, usize, usize), Vec<String>> =
                std::collections::HashMap::new();
            let mut expanded_goal_hashes: std::collections::HashSet<u64> =
                std::collections::HashSet::new();

            // LeanTree-ish state-key caches (goal-state based, not text based).
            const UNKNOWN_STATE_KEY: u64 = u64::MAX;
            let mut lean_state_cache_hits: usize = 0;
            let mut lean_state_cache_misses: usize = 0;
            let mut lean_state_candidates_cache: std::collections::HashMap<u64, Vec<String>> =
                std::collections::HashMap::new();
            // State-action outcomes: avoid re-trying the same tactic on the same goal-state.
            // Values are a small score where lower is better (more promising).
            let mut state_action_cache: std::collections::HashMap<(u64, u64), i32> =
                std::collections::HashMap::new(); // ((state_key, cand_hash) -> score)
                                                  // Cheap goal-dump cache for picking goals (keyed by (text_hash,len,focus_line)).
            let mut goal_dump_calls: usize = 0;
            let mut goal_dump_cache_hits: usize = 0;
            let mut goal_dump_cache_misses: usize = 0;
            let mut goal_dump_cache: std::collections::HashMap<
                (u64, usize, usize),
                (u64, usize, usize, String),
            > = std::collections::HashMap::new(); // (state_key, n_goals, hyps_total, target)
                                                  // Companion cache for `hyps_texts` (used by SMT ranking and tactic reranking).
            let mut goal_dump_hyps_cache_hits: usize = 0;
            let mut goal_dump_hyps_cache_misses: usize = 0;
            let mut goal_dump_hyps_cache: std::collections::HashMap<
                (u64, usize, usize),
                Vec<String>,
            > = std::collections::HashMap::new(); // (text_hash,len,line) -> hyps_texts
                                                  // Interpretation of these counters:
                                                  // - “hits” means we got `hyps_texts` without calling Lean (memory or disk).
                                                  // - “misses” means we *wanted* `hyps_texts` (for SMT/ranking) but couldn't find them.
                                                  // This is meant to track whether our “no extra Lean calls” design is actually working.

            // If we already have a goal snapshot for the initial focus, pre-populate the caches.
            // This enables state-aware ranking (and SMT precheck) even when we never invoke the
            // Lean oracle candidate generator.
            if let Some(gd) = goal_dump_v.as_ref() {
                if let Some(pp) = gd.get("pp_dump") {
                    if let Some(line) = gd
                        .get("selected_sorry")
                        .and_then(|s| s.get("line"))
                        .and_then(|v| v.as_u64())
                    {
                        let text_hash = hash_text(&original_text);
                        let key = (text_hash, original_text.len(), line as usize);
                        let state_key = hash_state_key(pp).unwrap_or(UNKNOWN_STATE_KEY);
                        let n_goals = pp
                            .get("goals")
                            .and_then(|v| v.as_array())
                            .map(|a| a.len())
                            .unwrap_or(0);
                        let hyps_total = pp
                            .get("goals")
                            .and_then(|v| v.as_array())
                            .map(|a| {
                                a.iter()
                                    .map(|g| {
                                        g.get("hyps")
                                            .and_then(|h| h.as_array())
                                            .map(|x| x.len())
                                            .unwrap_or(0)
                                    })
                                    .sum::<usize>()
                            })
                            .unwrap_or(0);
                        let target = pp
                            .get("goals")
                            .and_then(|v| v.as_array())
                            .and_then(|a| a.first())
                            .and_then(|g| g.get("pretty"))
                            .and_then(|v| v.as_str())
                            .and_then(|s| {
                                s.lines().find_map(|ln| {
                                    ln.trim_start()
                                        .strip_prefix("⊢")
                                        .map(|r| r.trim().to_string())
                                })
                            })
                            .unwrap_or_default();
                        goal_dump_cache.insert(key, (state_key, n_goals, hyps_total, target));

                        let hyps_texts: Vec<String> = pp
                            .get("goals")
                            .and_then(|v| v.as_array())
                            .and_then(|a| a.first())
                            .and_then(|g| g.get("hyps"))
                            .and_then(|v| v.as_array())
                            .map(|hs| {
                                hs.iter()
                                    .filter_map(|h| h.get("text").and_then(|v| v.as_str()))
                                    .map(|s| s.to_string())
                                    .collect()
                            })
                            .unwrap_or_default();
                        goal_dump_hyps_cache.insert(key, hyps_texts);
                    }
                }
            }

            #[cfg(feature = "planner")]
            let mut planner_cache: std::collections::HashMap<
                (u64, u64),
                plc::planner::PlannerDecision,
            > = std::collections::HashMap::new(); // ((state_key, goal_sig) -> decision)
            #[cfg(feature = "planner")]
            let mut planner_cache_hits: u64 = 0;
            #[cfg(feature = "planner")]
            let mut planner_cache_misses: u64 = 0;
            #[cfg(feature = "planner")]
            let mut prof_planner_ms: u64 = 0;

            let mut smt_entails_cache: std::collections::HashMap<(u64, u64, usize), bool> =
                std::collections::HashMap::new(); // ((state_key, goal_sig, depth) -> entails)
            let mut smt_artifacts_done: std::collections::HashSet<(u64, u64, usize)> =
                std::collections::HashSet::new(); // ((state_key, goal_sig, depth) -> already produced artifacts)
            let mut smt_proof_done: std::collections::HashSet<(u64, u64, usize)> =
                std::collections::HashSet::new(); // ((state_key, goal_sig, depth) -> attempted proof capture)
            let mut smt_proof_dump_done: std::collections::HashSet<(u64, u64, usize)> =
                std::collections::HashSet::new(); // ((state_key, goal_sig, depth) -> attempted proof dump capture)
            let mut smt_cache_hits: u64 = 0;
            let mut smt_cache_misses: u64 = 0;
            let mut smt_errors: u64 = 0;
            let mut smt_last_error: Option<String> = None;
            let mut smt_dumps_written: u64 = 0;
            let mut smt_dump_paths: Vec<String> = Vec::new();
            // Keep a bounded copy of the most recent SMT dump so JSON consumers can see it even if
            // opening the dump file is inconvenient.
            let mut smt_dump_last_path: Option<String> = None;
            let mut smt_dump_last_chars: Option<usize> = None;
            let mut smt_dump_last_preview: Option<String> = None;
            let mut smt_proof_attempts: u64 = 0;
            let mut smt_proofs_captured: u64 = 0;
            let mut smt_proof_last_error: Option<String> = None;
            let mut smt_proof_last: Option<serde_json::Value> = None;
            let mut smt_proof_dump_paths: Vec<String> = Vec::new();
            let mut smt_proof_dump_attempts: u64 = 0;
            let mut smt_proof_dump_written: u64 = 0;
            let mut smt_proof_dump_skipped_too_large: u64 = 0;
            let mut smt_proof_dump_last_error: Option<String> = None;
            let mut smt_entails_attempts: u64 = 0;
            let mut smt_entails_escalations: u64 = 0;
            let mut smt_entails_trace: Vec<serde_json::Value> = Vec::new();
            let mut prof_smt_ms: u64 = 0;
            let mut smt_reuse: Option<plc::smt_lia::ReusableSmtSession> = None;

            // Best-effort: if the user asked for SMT proofs/dumps, sometimes we want to attempt
            // once directly from the initial goal dump (if present).
            //
            // Primary motivation: avoid "SMT entailment served from cache so we never recomputed
            // a proof object / dump artifact" surprises.
            //
            // We intentionally do **not** do this unconditionally, because if SMT entailment is
            // computed freshly (cache miss), the later path can produce the same artifacts and we'd
            // double-count / duplicate files.
            if (smt_proof || smt_proof_dump)
                && (smt_proof_attempts == 0 || smt_proof_dump_attempts == 0)
            {
                if let Some(gd) = goal_dump_v.as_ref() {
                    if let Some(pp) = gd.get("pp_dump") {
                        let state_key = hash_state_key(pp).unwrap_or(UNKNOWN_STATE_KEY);
                        let target = pp
                            .get("goals")
                            .and_then(|v| v.as_array())
                            .and_then(|a| a.first())
                            .and_then(|g| g.get("pretty"))
                            .and_then(|v| v.as_str())
                            .and_then(|s| {
                                s.lines().find_map(|ln| {
                                    ln.trim_start()
                                        .strip_prefix("⊢")
                                        .map(|r| r.trim().to_string())
                                })
                            })
                            .unwrap_or_default();
                        let goal_sig = hash_text(&target);

                        if state_key != UNKNOWN_STATE_KEY && !target.is_empty() {
                            let eager_ok = if !smt_precheck {
                                true
                            } else if let Some(cd) = cache_dir.as_ref() {
                                cache_read_smt_entails(cd, state_key, goal_sig, smt_depth).is_some()
                            } else {
                                false
                            };

                            if !eager_ok {
                                // Expect a cache miss (or no SMT precheck), so let the normal
                                // computation path drive artifact generation.
                                // (We still keep the `--smt-proof-dump` directory creation behavior elsewhere.)
                            } else {
                                let ck = (state_key, goal_sig, smt_depth);

                                if smt_proof && smt_proof_done.insert(ck) {
                                    smt_proof_attempts = smt_proof_attempts.saturating_add(1);
                                    match plc::smt_lia::unsat_proof_from_pp_dump(
                                        pp,
                                        smt_timeout_ms,
                                        smt_seed,
                                        smt_depth,
                                        smt_proof_max_chars,
                                    ) {
                                        Ok(Some(pf)) => {
                                            smt_proofs_captured =
                                                smt_proofs_captured.saturating_add(1);
                                            smt_proof_last = Some(pf);
                                        }
                                        Ok(None) => {
                                            smt_proof_last_error =
                                                Some("proof_unavailable".to_string());
                                        }
                                        Err(e) => {
                                            smt_proof_last_error = Some(truncate_str(&e, 400));
                                        }
                                    }
                                }

                                if smt_proof && smt_proof_dump && smt_proof_dump_done.insert(ck) {
                                    smt_proof_dump_attempts =
                                        smt_proof_dump_attempts.saturating_add(1);
                                    match plc::smt_lia::unsat_proof_from_pp_dump(
                                        pp,
                                        smt_timeout_ms,
                                        smt_seed,
                                        smt_depth,
                                        smt_proof_dump_max_chars,
                                    ) {
                                        Ok(Some(pf_full)) => {
                                            let preview = pf_full
                                                .get("preview")
                                                .and_then(|v| v.as_str())
                                                .unwrap_or("");
                                            let total_chars = pf_full
                                                .get("chars")
                                                .and_then(|v| v.as_u64())
                                                .unwrap_or(0)
                                                as usize;
                                            let preview_chars = preview.chars().count();
                                            if total_chars <= smt_proof_dump_max_chars
                                                && preview_chars == total_chars
                                            {
                                                let root = smt_proof_dump_root(
                                                    &repo_root,
                                                    &smt_proof_dump_dir_opt,
                                                );
                                                let path = maybe_write_smt_proof_dump(
                                                    &root, state_key, goal_sig, smt_depth, preview,
                                                );
                                                if !smt_proof_dump_paths.contains(&path) {
                                                    smt_proof_dump_paths.push(path);
                                                    smt_proof_dump_written =
                                                        smt_proof_dump_written.saturating_add(1);
                                                }
                                            } else {
                                                smt_proof_dump_skipped_too_large =
                                                    smt_proof_dump_skipped_too_large
                                                        .saturating_add(1);
                                                smt_proof_dump_last_error = Some(format!(
                                                "proof_too_large_for_dump chars={total_chars} max_chars={}",
                                                smt_proof_dump_max_chars
                                            ));
                                            }
                                        }
                                        Ok(None) => {
                                            smt_proof_dump_last_error =
                                                Some("proof_unavailable".to_string());
                                        }
                                        Err(e) => {
                                            smt_proof_dump_last_error = Some(truncate_str(&e, 400));
                                        }
                                    }
                                }

                                // Mark artifacts as produced for this key so cache-hit paths can skip
                                // redundant proof/dump generation later in the run.
                                smt_artifacts_done.insert((state_key, goal_sig, smt_depth));
                            }
                        }
                    }
                }
            }

            // Baseline verify (for first-error line; also returned in output).
            let (baseline_raw_v, baseline_summary, baseline_ms, baseline_skipped) =
                if let Some(dur) = budget_dur(timeout_s) {
                    let t0 = std::time::Instant::now();
                    let baseline = rt
                        .block_on(plc::verify_lean_file(&repo_root, &file, dur))
                        .map_err(|e| format!("verify failed: {e}"))?;
                    let baseline_ms = t0.elapsed().as_millis() as u64;
                    prof_verify_baseline_ms = prof_verify_baseline_ms.saturating_add(baseline_ms);
                    prof_verify_baseline_calls += 1;
                    let baseline_raw_v = serde_json::to_value(baseline)
                        .map_err(|e| format!("serialize verify: {e}"))?;
                    let baseline_summary = verify_summary_from_raw_value(&baseline_raw_v);
                    (baseline_raw_v, baseline_summary, baseline_ms, false)
                } else {
                    bailed_total_timeout = true;
                    record_event(
                        "bailout_total_timeout",
                        json!({ "where": "baseline_verify" }),
                    );
                    let raw = plc::VerifyResult {
                        ok: false,
                        timeout: true,
                        returncode: None,
                        stdout: String::new(),
                        stderr: "total timeout (budget exhausted before baseline verify)"
                            .to_string(),
                        cmd: vec![],
                        cwd: repo_root.display().to_string(),
                        tmp_file: None,
                    };
                    let raw_v =
                        serde_json::to_value(raw).map_err(|e| format!("serialize verify: {e}"))?;
                    let summary = verify_summary_from_raw_value(&raw_v);
                    (raw_v, summary, 0u64, true)
                };
            record_event(
                "baseline_verify",
                json!({
                    "skipped": baseline_skipped,
                    "ms": if baseline_skipped { serde_json::Value::Null } else { json!(baseline_ms) },
                    "summary": baseline_summary,
                }),
            );

            // Establish a focus declaration for the whole search (best-effort).
            // This makes depth>1 behave like a proof-tree search: keep patching within the same lemma.
            //
            // If the user provided `--focus-line`, use the `sorry` closest to that line as the
            // focus decl/line (otherwise we tend to get stuck on the first `sorry` in the file).
            let (
                focus_decl_name,
                focus_line_1,
                focus_sorry,
                focus_source,
                focus_requested_decl,
                focus_available_decls,
            ) = {
                let locs0 = plc::locate_sorries_in_text(&original_text, 200, 1).unwrap_or_default();
                let available: Vec<String> = locs0
                    .iter()
                    .filter_map(|l| l.decl_name.clone())
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();
                let mut available = available;
                available.sort();
                let available_short: Vec<String> = available.into_iter().take(24).collect();
                if let Some(fd) = focus_decl_override.clone() {
                    let fd = fd.trim().to_string();
                    let fd_last = fd
                        .split(|c| c == '.' || c == ':')
                        .filter(|s| !s.is_empty())
                        .last()
                        .unwrap_or(fd.as_str())
                        .to_string();
                    let picked_exact = locs0
                        .iter()
                        .find(|l| l.decl_name.as_deref() == Some(fd.as_str()))
                        .cloned();
                    // Best-effort unqualified match:
                    // - some locators only record the local name (`foo`) rather than `Namespace.foo`
                    let needle = format!(".{fd_last}");
                    let picked_suffix = locs0
                        .iter()
                        .find(|l| {
                            l.decl_name
                                .as_ref()
                                .is_some_and(|dn| dn.ends_with(&needle) || dn == &fd_last)
                        })
                        .cloned();
                    let picked = picked_exact.or(picked_suffix);
                    if picked.is_some() {
                        (
                            picked.as_ref().and_then(|s| s.decl_name.clone()),
                            picked.as_ref().map(|s| s.line),
                            picked,
                            "focus_decl",
                            Some(fd),
                            available_short,
                        )
                    } else {
                        let first_error_line_1 = baseline_summary
                            .get("first_error_loc")
                            .and_then(|l| l.get("line"))
                            .and_then(|v| v.as_u64())
                            .map(|x| x as usize);
                        let picked = plc::select_primary_sorry(first_error_line_1, &locs0);
                        (
                            picked.as_ref().and_then(|s| s.decl_name.clone()),
                            picked.as_ref().map(|s| s.line),
                            picked,
                            "focus_decl_not_found_fallback",
                            Some(fd),
                            available_short,
                        )
                    }
                } else if let Some(fl) = focus_line_override {
                    let picked = locs0
                        .iter()
                        .min_by_key(|l| (l.line as i64 - fl as i64).abs())
                        .cloned();
                    (
                        picked.as_ref().and_then(|s| s.decl_name.clone()),
                        picked.as_ref().map(|s| s.line),
                        picked,
                        "focus_line",
                        None,
                        available_short,
                    )
                } else {
                    let first_error_line_1 = baseline_summary
                        .get("first_error_loc")
                        .and_then(|l| l.get("line"))
                        .and_then(|v| v.as_u64())
                        .map(|x| x as usize);
                    let picked = plc::select_primary_sorry(first_error_line_1, &locs0);
                    (
                        picked.as_ref().and_then(|s| s.decl_name.clone()),
                        picked.as_ref().map(|s| s.line),
                        picked,
                        "first_error_or_primary_sorry",
                        None,
                        available_short,
                    )
                }
            };
            record_event(
                "focus",
                json!({
                    "decl": focus_decl_name,
                    "line": focus_line_1,
                    "source": focus_source,
                    "requested_decl": focus_requested_decl,
                    "available_decls": focus_available_decls,
                }),
            );

            // Note: hard focus mismatch is handled earlier (before baseline verify).

            // Small helper for grokkable reporting.
            let classify_failure_mode = |first_error: Option<&str>| -> &'static str {
                let s = first_error.unwrap_or("").to_lowercase();
                if s.contains("unknown tactic") {
                    "unknown_tactic"
                } else if s.contains("synthinstancefailed")
                    || s.contains("failed to synthesize instance")
                {
                    "typeclass_instance_failed"
                } else if s.contains("unsolved goals") {
                    "unsolved_goals"
                } else if s.contains("omega could not prove") || s.contains("no usable constraints")
                {
                    "omega_failed"
                } else if s.contains("linarith failed") {
                    "linarith_failed"
                } else if s.contains("made no progress") {
                    "made_no_progress"
                } else if s.contains("error") {
                    "lean_error"
                } else {
                    "none"
                }
            };

            // Aggregate failure modes across node evaluations (best-effort, bounded).
            // These are for grokability only; they must not affect control flow.
            let _failure_modes_nodes: std::collections::HashMap<String, u64> =
                std::collections::HashMap::new();

            let mut next_id = 1usize;
            let mut all: Vec<Node> = Vec::new();
            let mut eval_cache: std::collections::HashMap<u64, CachedEval> =
                std::collections::HashMap::new();
            let mut disk_cache_eval_hits: u64 = 0;
            let mut disk_cache_eval_misses: u64 = 0;
            // Always compute root sorry counts so early bailouts still have meaningful output.
            let root_sorries = plc::locate_sorries_in_text(&original_text, 500, 1)
                .unwrap_or_default()
                .len();
            let root_conservative_sorries =
                plc::count_sorry_tokens_conservative(&original_text).unwrap_or(0);
            let mut frontier: Vec<Node> = vec![Node {
                id: 0,
                depth: 0,
                text: original_text.clone(),
                focus_decl_name: focus_decl_name.clone(),
                focus_line: focus_line_1,
                focus_goal_sig: None,
                last_region: None,
                last_replacement: None,
                parent_id: None,
                verify_raw: Some(baseline_raw_v.clone()),
                verify_summary: Some(baseline_summary.clone()),
                sorries: Some(root_sorries),
                conservative_sorries: Some(root_conservative_sorries),
                smt_hint: None,
                rank_hint: None,
            }];

            let mut best_done: Option<Node> = None;

            while !frontier.is_empty() && all.len() < max_nodes {
                if std::time::Instant::now() >= run_deadline {
                    bailed_total_timeout = true;
                    record_event("bailout_total_timeout", json!({ "where": "main_loop" }));
                    break;
                }
                // Evaluate current frontier nodes if needed; then expand best-first by score.
                for n in frontier.iter_mut() {
                    if std::time::Instant::now() >= run_deadline {
                        bailed_total_timeout = true;
                        record_event(
                            "bailout_total_timeout",
                            json!({ "where": "frontier_prefill" }),
                        );
                        break;
                    }
                    // Disk cache (eval) prefill: if we have the full eval for this text, it supplies
                    // verify + sorry counts in one shot and avoids redundant work.
                    if (n.verify_summary.is_none() || n.sorries.is_none())
                        && n.verify_raw.is_some() == false
                    {
                        if let Some(cd) = cache_dir.as_ref() {
                            let h = hash_text(&n.text);
                            if let Some((raw_v, summary, sorries, conservative)) =
                                cache_read_eval(cd, h, n.text.len())
                            {
                                disk_cache_eval_hits += 1;
                                n.verify_raw = Some(raw_v.clone());
                                n.verify_summary = Some(summary.clone());
                                n.sorries = Some(sorries);
                                n.conservative_sorries = Some(conservative);
                                eval_cache.insert(
                                    h,
                                    CachedEval {
                                        len: n.text.len(),
                                        verify_raw: raw_v,
                                        verify_summary: summary,
                                        sorries,
                                        conservative_sorries: conservative,
                                    },
                                );
                            } else {
                                disk_cache_eval_misses += 1;
                            }
                        }
                    }
                    if n.sorries.is_none() {
                        let h = hash_text(&n.text);
                        if let Some(c) = eval_cache.get(&h).filter(|c| c.len == n.text.len()) {
                            n.sorries = Some(c.sorries);
                            n.conservative_sorries = Some(c.conservative_sorries);
                        } else {
                            let t0 = std::time::Instant::now();
                            let locs =
                                plc::locate_sorries_in_text(&n.text, 500, 1).unwrap_or_default();
                            prof_locate_sorries_ms = prof_locate_sorries_ms
                                .saturating_add(t0.elapsed().as_millis() as u64);
                            n.sorries = Some(locs.len());
                            let t0 = std::time::Instant::now();
                            n.conservative_sorries =
                                Some(plc::count_sorry_tokens_conservative(&n.text).unwrap_or(0));
                            prof_conservative_sorries_ms = prof_conservative_sorries_ms
                                .saturating_add(t0.elapsed().as_millis() as u64);
                        }
                    }
                    if n.verify_summary.is_none() {
                        let h = hash_text(&n.text);
                        if let Some(c) = eval_cache.get(&h).filter(|c| c.len == n.text.len()) {
                            n.verify_raw = Some(c.verify_raw.clone());
                            n.verify_summary = Some(c.verify_summary.clone());
                        } else {
                            if let Some(cd) = cache_dir.as_ref() {
                                if let Some((raw_v, summary, sorries, conservative)) =
                                    cache_read_eval(cd, h, n.text.len())
                                {
                                    disk_cache_eval_hits += 1;
                                    eval_cache.insert(
                                        h,
                                        CachedEval {
                                            len: n.text.len(),
                                            verify_raw: raw_v.clone(),
                                            verify_summary: summary.clone(),
                                            sorries,
                                            conservative_sorries: conservative,
                                        },
                                    );
                                    n.verify_raw = Some(raw_v);
                                    n.verify_summary = Some(summary);
                                    n.sorries = Some(sorries);
                                    n.conservative_sorries = Some(conservative);
                                    continue;
                                } else {
                                    disk_cache_eval_misses += 1;
                                }
                            }
                            let Some(dur) = budget_dur(timeout_s) else {
                                bailed_total_timeout = true;
                                record_event(
                                    "bailout_total_timeout",
                                    json!({ "where": "frontier_verify" }),
                                );
                                break;
                            };
                            let t0 = std::time::Instant::now();
                            let raw = rt
                                .block_on(plc::verify_lean_text(&repo_root, &n.text, dur))
                                .map_err(|e| format!("verify failed: {e}"))?;
                            prof_verify_nodes_ms = prof_verify_nodes_ms
                                .saturating_add(t0.elapsed().as_millis() as u64);
                            prof_verify_nodes_calls += 1;
                            let raw_v = serde_json::to_value(raw)
                                .map_err(|e| format!("serialize verify: {e}"))?;
                            let summary = verify_summary_from_raw_value(&raw_v);
                            let sorries = n.sorries.unwrap_or(999);
                            let conservative = n.conservative_sorries.unwrap_or(999);
                            eval_cache.insert(
                                h,
                                CachedEval {
                                    len: n.text.len(),
                                    verify_raw: raw_v.clone(),
                                    verify_summary: summary.clone(),
                                    sorries,
                                    conservative_sorries: conservative,
                                },
                            );
                            if let Some(cd) = cache_dir.as_ref() {
                                cache_write_eval(
                                    cd,
                                    h,
                                    n.text.len(),
                                    &raw_v,
                                    &summary,
                                    sorries,
                                    conservative,
                                );
                            }
                            n.verify_raw = Some(raw_v);
                            n.verify_summary = Some(summary);
                        }
                    }

                    // Success condition: ok + no remaining `locate` sorries.
                    let ok = n
                        .verify_summary
                        .as_ref()
                        .and_then(|s| s.get("ok"))
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let sorry_warnings = n
                        .verify_summary
                        .as_ref()
                        .and_then(|s| s.get("counts"))
                        .and_then(|c| c.get("sorry_warnings"))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let sorries = n.sorries.unwrap_or(999);
                    if ok && sorries == 0 && sorry_warnings == 0 {
                        best_done = Some(n.clone());
                        break;
                    }
                }
                if bailed_total_timeout {
                    break;
                }
                if best_done.is_some() {
                    break;
                }

                // Sort frontier by score.
                frontier.sort_by(|a, b| {
                    let sa = a.verify_summary.as_ref().unwrap();
                    let sb = b.verify_summary.as_ref().unwrap();
                    let mut ka = verify_score_key(
                        sa,
                        a.sorries.unwrap_or(999),
                        a.conservative_sorries.unwrap_or(999),
                    );
                    let mut kb = verify_score_key(
                        sb,
                        b.sorries.unwrap_or(999),
                        b.conservative_sorries.unwrap_or(999),
                    );
                    if depth_bonus > 0 {
                        // Lower is better; subtracting rewards deeper nodes slightly (tie-break/escape hatch).
                        ka.1 =
                            ka.1.saturating_sub(depth_bonus.saturating_mul(a.depth as i64));
                        kb.1 =
                            kb.1.saturating_sub(depth_bonus.saturating_mul(b.depth as i64));
                    }
                    ka.cmp(&kb).then_with(|| a.id.cmp(&b.id))
                });

                // Keep top beam, but also keep one “best progress” node even if it would be dropped.
                if frontier.len() > beam {
                    let mut best_progress_idx = 0usize;
                    let mut best_progress_key = (i64::MAX, 9, i64::MAX, i64::MAX);
                    for (i, n) in frontier.iter().enumerate() {
                        let s = n.verify_summary.as_ref().unwrap();
                        let k = progress_score_key(
                            s,
                            n.sorries.unwrap_or(999),
                            n.conservative_sorries.unwrap_or(999),
                        );
                        if k < best_progress_key {
                            best_progress_key = k;
                            best_progress_idx = i;
                        }
                    }
                    let progress_node = frontier[best_progress_idx].clone();
                    frontier.truncate(beam);
                    if frontier.iter().all(|n| n.id != progress_node.id) && beam > 0 {
                        // Replace the last slot with the progress node (beam size stays constant).
                        *frontier.last_mut().expect("beam > 0 implies non-empty") = progress_node;
                    }
                }

                // Move current frontier nodes into all (trace).
                for n in frontier.iter() {
                    all.push(n.clone());
                    if all.len() >= max_nodes {
                        break;
                    }
                }
                if all.len() >= max_nodes {
                    break;
                }

                // Expand.
                let mut new_frontier: Vec<Node> = Vec::new();
                for parent in frontier.iter() {
                    if std::time::Instant::now() >= run_deadline {
                        bailed_total_timeout = true;
                        record_event("bailout_total_timeout", json!({ "where": "expand" }));
                        break;
                    }
                    if parent.depth >= depth {
                        continue;
                    }

                    // Pick next sorry to patch in this node.
                    let locs_all =
                        plc::locate_sorries_in_text(&parent.text, 200, 1).unwrap_or_default();
                    let locs = if let Some(dn) = parent.focus_decl_name.as_deref() {
                        // Decl name matching can be inconsistent across locators:
                        // sometimes we get `foo`, sometimes `Namespace.foo`.
                        // Treat them as equivalent by comparing the last segment too.
                        let dn_last = dn
                            .split(|c| c == '.' || c == ':')
                            .filter(|s| !s.is_empty())
                            .last()
                            .unwrap_or(dn);
                        let needle = format!(".{dn_last}");
                        let xs: Vec<plc::SorryLocation> = locs_all
                            .iter()
                            .cloned()
                            .filter(|l| {
                                l.decl_name.as_ref().is_some_and(|got| {
                                    got == dn || got == dn_last || got.ends_with(&needle)
                                })
                            })
                            .collect();
                        if xs.is_empty() {
                            if focus_decl_hard {
                                // Hard focus: do not drift to other declarations.
                                record_event(
                                    "focus_decl_hard_no_sorries_in_decl",
                                    json!({ "decl": dn, "node_id": parent.id }),
                                );
                                vec![]
                            } else {
                                locs_all
                            }
                        } else {
                            xs
                        }
                    } else {
                        locs_all
                    };
                    // Optional goal-branch stabilization: if we know the goal signature for the current branch,
                    // prefer holes whose cached goal target matches it (cache-first; no extra Lean calls).
                    let locs = if let Some(gs) = parent.focus_goal_sig {
                        let th = hash_text(&parent.text);
                        let mut xs: Vec<plc::SorryLocation> = Vec::new();
                        for l in locs.iter() {
                            let k = (th, parent.text.len(), l.line);
                            // In-memory target from goal-dump cache.
                            let mut goal_sig_opt =
                                goal_dump_cache.get(&k).and_then(|(_, _, _, tgt)| {
                                    if tgt.is_empty() {
                                        None
                                    } else {
                                        Some(hash_text(tgt))
                                    }
                                });
                            // Disk fallback (still cache-only).
                            if goal_sig_opt.is_none() {
                                if let Some(cd) = cache_dir.as_ref() {
                                    if let Some(tup) =
                                        cache_read_goal_dump(cd, th, parent.text.len(), l.line)
                                    {
                                        let (_sk, _ng, _ht, tgt) = tup.clone();
                                        if !tgt.is_empty() {
                                            goal_sig_opt = Some(hash_text(&tgt));
                                            // Keep the cached state_key/metrics intact (don’t downgrade to UNKNOWN).
                                            goal_dump_cache.insert(k, tup);
                                        }
                                    }
                                }
                            }
                            if goal_sig_opt == Some(gs) {
                                xs.push(l.clone());
                            }
                        }
                        if xs.is_empty() {
                            locs
                        } else {
                            xs
                        }
                    } else {
                        locs
                    };
                    let first_error_line_1 = parent
                        .verify_summary
                        .as_ref()
                        .and_then(|s| s.get("first_error_loc"))
                        .and_then(|l| l.get("line"))
                        .and_then(|v| v.as_u64())
                        .map(|x| x as usize);
                    if locs.is_empty() {
                        // Hard-focus leaf: nothing to patch in the target decl for this node.
                        continue;
                    }
                    let selected = if candidates_mode == "lean-try"
                        && goal_first_k > 0
                        && !locs.is_empty()
                    {
                        // Goal-first scheduler (bounded): probe a few candidate holes with a cheap goal dump,
                        // compute a state key and a crude "difficulty" score, then pick the easiest.
                        let fl = parent
                            .focus_line
                            .or(first_error_line_1)
                            .unwrap_or(locs[0].line);
                        let mut cands: Vec<plc::SorryLocation> = locs.clone();
                        cands.sort_by_key(|l| (l.line as i64 - fl as i64).abs());
                        cands.truncate(goal_first_k);

                        // Optional LLM planner: can override which hole to focus and how much oracle budget to spend.
                        // This is feature-gated and defaults off.
                        let planner_selected: Option<plc::SorryLocation> = if llm_planner {
                            #[cfg(not(feature = "planner"))]
                            {
                                return Err(
                                    "--llm-planner requires building with cargo feature `planner`"
                                        .to_string(),
                                );
                            }
                            #[cfg(feature = "planner")]
                            {
                                let mut picked_sel: Option<plc::SorryLocation> = None;

                                // Probe only the closest hole to build planner evidence.
                                let seed =
                                    cands.first().cloned().unwrap_or_else(|| locs[0].clone());
                                let th = hash_text(&parent.text);
                                let key = (th, parent.text.len(), seed.line);

                                // Ensure we have (state_key,n_goals,hyps_total,target) for the seed.
                                let mut target: String = String::new();
                                let (state_key, n_goals, hyps_total, target_cached) =
                                    if let Some(v) = goal_dump_cache.get(&key) {
                                        v.clone()
                                    } else {
                                        if lean_oracle_calls >= lean_oracle_max_calls {
                                            // No budget to probe; skip planner.
                                            (UNKNOWN_STATE_KEY, 0usize, 0usize, String::new())
                                        } else {
                                            goal_dump_calls += 1;
                                            lean_oracle_calls += 1;
                                            let t0 = std::time::Instant::now();
                                            let gd = if let Some(dur) =
                                                budget_dur(goal_dump_timeout_s)
                                            {
                                                rt.block_on(plc::goal_dump_in_text_at(
                                                    &repo_root,
                                                    &file,
                                                    &parent.text,
                                                    dur,
                                                    Some(seed.line),
                                                    first_error_line_1,
                                                ))
                                                .ok()
                                            } else {
                                                bailed_total_timeout = true;
                                                record_event(
                                                    "bailout_total_timeout",
                                                    json!({ "where": "goal_dump_planner_seed" }),
                                                );
                                                None
                                            };
                                            let elapsed_ms = t0.elapsed().as_millis() as u64;
                                            prof_goal_dump_ms =
                                                prof_goal_dump_ms.saturating_add(elapsed_ms);

                                            let pp = gd.as_ref().and_then(|v| v.get("pp_dump"));
                                            let state_key = pp
                                                .and_then(|pp| hash_state_key(pp))
                                                .unwrap_or(UNKNOWN_STATE_KEY);
                                            let n_goals = pp
                                                .and_then(|pp| pp.get("goals"))
                                                .and_then(|v| v.as_array())
                                                .map(|a| a.len())
                                                .unwrap_or(0);
                                            let hyps_total = pp
                                                .and_then(|pp| pp.get("goals"))
                                                .and_then(|v| v.as_array())
                                                .map(|a| {
                                                    a.iter()
                                                        .map(|g| {
                                                            g.get("hyps")
                                                                .and_then(|h| h.as_array())
                                                                .map(|x| x.len())
                                                                .unwrap_or(0)
                                                        })
                                                        .sum::<usize>()
                                                })
                                                .unwrap_or(0);
                                            target = pp
                                                .and_then(|pp| pp.get("goals"))
                                                .and_then(|v| v.as_array())
                                                .and_then(|a| a.first())
                                                .and_then(|g| g.get("pretty"))
                                                .and_then(|v| v.as_str())
                                                .and_then(|s| {
                                                    s.lines().find_map(|ln| {
                                                        ln.trim_start()
                                                            .strip_prefix("⊢")
                                                            .map(|r| r.trim().to_string())
                                                    })
                                                })
                                                .unwrap_or_default();
                                            let hyps_texts: Vec<String> = pp
                                                .and_then(|pp| pp.get("goals"))
                                                .and_then(|v| v.as_array())
                                                .and_then(|a| a.first())
                                                .and_then(|g| g.get("hyps"))
                                                .and_then(|v| v.as_array())
                                                .map(|a| {
                                                    a.iter()
                                                        .filter_map(|h| {
                                                            h.get("text")
                                                                .and_then(|v| v.as_str())
                                                                .map(|s| s.to_string())
                                                        })
                                                        .collect::<Vec<_>>()
                                                })
                                                .unwrap_or_default();
                                            let tup =
                                                (state_key, n_goals, hyps_total, target.clone());
                                            goal_dump_cache.insert(key, tup.clone());
                                            goal_dump_hyps_cache.insert(key, hyps_texts.clone());
                                            if let Some(cd) = cache_dir.as_ref() {
                                                cache_write_goal_dump(
                                                    cd,
                                                    th,
                                                    parent.text.len(),
                                                    seed.line,
                                                    state_key,
                                                    n_goals,
                                                    hyps_total,
                                                    &target,
                                                    &hyps_texts,
                                                );
                                            }
                                            tup
                                        }
                                    };

                                if target.is_empty() {
                                    target = target_cached;
                                }
                                let goal_sig = hash_text(&target);
                                let cache_key = (state_key, goal_sig);

                                let decision = if let Some(d) =
                                    planner_cache.get(&cache_key).cloned()
                                {
                                    planner_cache_hits += 1;
                                    Some(d)
                                } else if let Some(cd) = cache_dir.as_ref() {
                                    if let Some(v) = cache_read_planner(cd, state_key, goal_sig) {
                                        if let Ok(d) =
                                            serde_json::from_value::<plc::planner::PlannerDecision>(
                                                v.clone(),
                                            )
                                        {
                                            planner_cache_hits += 1;
                                            planner_cache.insert(cache_key, d.clone());
                                            Some(d)
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                };

                                let decision = if let Some(d) = decision {
                                    d
                                } else {
                                    planner_cache_misses += 1;
                                    let evidence = plc::planner::PlannerEvidence {
                                        state_key,
                                        goal: plc::planner::PlannerGoalSummary {
                                            target: truncate_str(&target, 400),
                                            n_goals: n_goals as u64,
                                            hyps_total: hyps_total as u64,
                                        },
                                        candidate_holes: cands
                                            .iter()
                                            .map(|h| plc::planner::PlannerHole {
                                                line_1: h.line as u64,
                                                excerpt: truncate_str(&h.excerpt, 220),
                                            })
                                            .collect(),
                                        config: plc::planner::PlannerConfigSummary {
                                            goal_first_k: goal_first_k as u64,
                                            oracle_max_calls: lean_oracle_max_calls as u64,
                                            timeout_s,
                                        },
                                    };

                                    let system = r#"You are a proof-search planner for Lean 4.
Given JSON evidence about the current goal and nearby `sorry` holes, choose where to focus and how to spend oracle budget.

Return STRICT JSON (no markdown) matching this shape:
{
  "confidence": 0.0-1.0,
  "focus_line_1": number|null,
  "oracle_passes": number|null,
  "oracle_tactics": [string...],
  "ban_oracle_tactics": [string...],
  "rationale": string
}

Constraints:
- If you recommend a focus line, it MUST be one of candidate_holes.line_1.
- Prefer low oracle_passes unless high confidence.
- Keep oracle_tactics within {"simp?","exact?","apply?","aesop?"}.
- For inequality-heavy goals, ban "aesop?".
"#;
                                    let ev_json = serde_json::to_string(&evidence)
                                        .unwrap_or_else(|_| "{}".to_string());
                                    let t0 = std::time::Instant::now();
                                    let res = rt.block_on(plc::planner::plan(
                                        system,
                                        &ev_json,
                                        StdDuration::from_secs(llm_planner_timeout_s),
                                    ));
                                    prof_planner_ms = prof_planner_ms
                                        .saturating_add(t0.elapsed().as_millis() as u64);
                                    let d = match res {
                                        Ok((d, _raw)) => d,
                                        Err(_) => plc::planner::PlannerDecision {
                                            confidence: 0.0,
                                            focus_line_1: None,
                                            oracle_passes: None,
                                            oracle_tactics: vec![],
                                            ban_oracle_tactics: vec![],
                                            rationale: "planner_failed".to_string(),
                                        },
                                    };
                                    planner_cache.insert(cache_key, d.clone());
                                    if let Some(cd) = cache_dir.as_ref() {
                                        if let Ok(v) = serde_json::to_value(&d) {
                                            cache_write_planner(cd, state_key, goal_sig, &v);
                                        }
                                    }
                                    d
                                };

                                if decision.confidence >= 0.4 {
                                    if let Some(passes) = decision.oracle_passes {
                                        std::env::set_var(
                                            "PROOFPATCH_ORACLE_PASSES",
                                            passes.to_string(),
                                        );
                                    }
                                    if !decision.oracle_tactics.is_empty() {
                                        std::env::set_var(
                                            "PROOFPATCH_ORACLE_TACTICS",
                                            decision.oracle_tactics.join(","),
                                        );
                                    }
                                    if !decision.ban_oracle_tactics.is_empty() {
                                        std::env::set_var(
                                            "PROOFPATCH_ORACLE_BAN",
                                            decision.ban_oracle_tactics.join(","),
                                        );
                                    }
                                    if let Some(fl1) = decision.focus_line_1 {
                                        if let Some(picked) =
                                            cands.iter().find(|h| h.line as u64 == fl1).cloned()
                                        {
                                            picked_sel = Some(picked);
                                        }
                                    }
                                }

                                picked_sel
                            }
                        } else {
                            None
                        };

                        if planner_selected.is_some() {
                            planner_selected
                        } else {
                            let mut best: Option<(i64, plc::SorryLocation)> = None;
                            let goal_first_slow_ms = std::env::var("PROOFPATCH_GOAL_FIRST_SLOW_MS")
                                .ok()
                                .and_then(|s| s.trim().parse::<u64>().ok())
                                .unwrap_or(1200);
                            for s0 in cands {
                                let mut last_probe_ms: Option<u64> = None;
                                // Cache lookup: (text hash, len, focus_line)
                                let th = hash_text(&parent.text);
                                let key = (th, parent.text.len(), s0.line);
                                let (state_key, n_goals, hyps_total, _target) = if let Some(v) =
                                    goal_dump_cache.get(&key)
                                {
                                    goal_dump_cache_hits += 1;
                                    v.clone()
                                } else if let Some(cd) = cache_dir.as_ref() {
                                    if let Some(tup) =
                                        cache_read_goal_dump(cd, th, parent.text.len(), s0.line)
                                    {
                                        goal_dump_cache_hits += 1;
                                        goal_dump_cache.insert(key, tup.clone());
                                        if let Some(hyps) = cache_read_goal_dump_hyps_texts(
                                            cd,
                                            th,
                                            parent.text.len(),
                                            s0.line,
                                        ) {
                                            goal_dump_hyps_cache_hits += 1;
                                            goal_dump_hyps_cache.insert(key, hyps);
                                        } else {
                                            goal_dump_hyps_cache_misses += 1;
                                        }
                                        tup
                                    } else {
                                        goal_dump_cache_misses += 1;
                                        // Bounded: share budget with oracle calls.
                                        if lean_oracle_calls >= lean_oracle_max_calls {
                                            continue;
                                        }
                                        goal_dump_calls += 1;
                                        lean_oracle_calls += 1;
                                        let t0 = std::time::Instant::now();
                                        let gd = if let Some(dur) = budget_dur(goal_dump_timeout_s)
                                        {
                                            rt.block_on(plc::goal_dump_in_text_at(
                                                &repo_root,
                                                &file,
                                                &parent.text,
                                                dur,
                                                Some(s0.line),
                                                first_error_line_1,
                                            ))
                                            .ok()
                                        } else {
                                            bailed_total_timeout = true;
                                            record_event(
                                                "bailout_total_timeout",
                                                json!({ "where": "goal_dump_goal_first" }),
                                            );
                                            None
                                        };
                                        let elapsed_ms = t0.elapsed().as_millis() as u64;
                                        last_probe_ms = Some(elapsed_ms);
                                        prof_goal_dump_ms =
                                            prof_goal_dump_ms.saturating_add(elapsed_ms);
                                        let pp = gd.as_ref().and_then(|v| v.get("pp_dump"));
                                        let state_key = pp
                                            .and_then(|pp| hash_state_key(pp))
                                            .unwrap_or(UNKNOWN_STATE_KEY);
                                        let n_goals = pp
                                            .and_then(|pp| pp.get("goals"))
                                            .and_then(|v| v.as_array())
                                            .map(|a| a.len())
                                            .unwrap_or(0);
                                        let hyps_total = pp
                                            .and_then(|pp| pp.get("goals"))
                                            .and_then(|v| v.as_array())
                                            .map(|a| {
                                                a.iter()
                                                    .map(|g| {
                                                        g.get("hyps")
                                                            .and_then(|h| h.as_array())
                                                            .map(|x| x.len())
                                                            .unwrap_or(0)
                                                    })
                                                    .sum::<usize>()
                                            })
                                            .unwrap_or(0);
                                        let target = pp
                                            .and_then(|pp| pp.get("goals"))
                                            .and_then(|v| v.as_array())
                                            .and_then(|a| a.first())
                                            .and_then(|g| g.get("pretty"))
                                            .and_then(|v| v.as_str())
                                            .and_then(|s| {
                                                s.lines().find_map(|ln| {
                                                    ln.trim_start()
                                                        .strip_prefix("⊢")
                                                        .map(|r| r.trim().to_string())
                                                })
                                            })
                                            .unwrap_or_default();
                                        let hyps_texts: Vec<String> = pp
                                            .and_then(|pp| pp.get("goals"))
                                            .and_then(|v| v.as_array())
                                            .and_then(|a| a.first())
                                            .and_then(|g| g.get("hyps"))
                                            .and_then(|v| v.as_array())
                                            .map(|a| {
                                                a.iter()
                                                    .filter_map(|h| {
                                                        h.get("text")
                                                            .and_then(|v| v.as_str())
                                                            .map(|s| s.to_string())
                                                    })
                                                    .collect::<Vec<_>>()
                                            })
                                            .unwrap_or_default();
                                        let tup = (state_key, n_goals, hyps_total, target.clone());
                                        goal_dump_cache.insert(key, tup.clone());
                                        goal_dump_hyps_cache.insert(key, hyps_texts.clone());
                                        cache_write_goal_dump(
                                            cd,
                                            th,
                                            parent.text.len(),
                                            s0.line,
                                            state_key,
                                            n_goals,
                                            hyps_total,
                                            &target,
                                            &hyps_texts,
                                        );

                                        // SMT precheck: if hyps entail target (LIA-only), take this hole immediately.
                                        if smt_precheck {
                                            if let Some(pp0) = pp {
                                                let goal_sig = hash_text(&target);
                                                let ck = (state_key, goal_sig, smt_depth);
                                                let mut entails_opt: Option<bool> = None;
                                                if let Some(v) = smt_entails_cache.get(&ck).copied()
                                                {
                                                    smt_cache_hits += 1;
                                                    entails_opt = Some(v);
                                                } else if let Some(cd2) = cache_dir.as_ref() {
                                                    if let Some(v) = cache_read_smt_entails(
                                                        cd2, state_key, goal_sig, smt_depth,
                                                    ) {
                                                        smt_cache_hits += 1;
                                                        smt_entails_cache.insert(ck, v);
                                                        entails_opt = Some(v);
                                                    }
                                                }
                                                if entails_opt.is_none() {
                                                    smt_cache_misses += 1;
                                                    let t_smt0 = std::time::Instant::now();
                                                    let (entails, attempts) =
                                                        smt_entails_from_pp_dump_escalating(
                                                            pp0,
                                                            smt_timeout_ms,
                                                            smt_seed,
                                                            smt_depth,
                                                            &smt_solver_norm,
                                                            smt_aggressive,
                                                            &mut smt_reuse,
                                                            &mut smt_entails_trace,
                                                        )
                                                        .unwrap_or_else(|e| {
                                                            smt_errors += 1;
                                                            smt_last_error =
                                                                Some(truncate_str(&e, 400));
                                                            (None, 1)
                                                        });
                                                    smt_entails_attempts = smt_entails_attempts
                                                        .saturating_add(attempts);
                                                    if attempts > 1 {
                                                        smt_entails_escalations =
                                                            smt_entails_escalations.saturating_add(
                                                                attempts.saturating_sub(1),
                                                            );
                                                    }
                                                    prof_smt_ms = prof_smt_ms.saturating_add(
                                                        t_smt0.elapsed().as_millis() as u64,
                                                    );
                                                    if let Some(b) = entails {
                                                        smt_entails_cache.insert(ck, b);
                                                        if let Some(cd2) = cache_dir.as_ref() {
                                                            cache_write_smt_entails(
                                                                cd2, state_key, goal_sig,
                                                                smt_depth, b,
                                                            );
                                                        }
                                                        entails_opt = Some(b);
                                                    }
                                                }
                                                if entails_opt == Some(true) {
                                                    best = Some((-1, s0));
                                                    break;
                                                }
                                            }
                                        }
                                        tup
                                    }
                                } else {
                                    goal_dump_cache_misses += 1;
                                    // Bounded: share budget with oracle calls.
                                    if lean_oracle_calls >= lean_oracle_max_calls {
                                        continue;
                                    }
                                    goal_dump_calls += 1;
                                    lean_oracle_calls += 1;
                                    let t0 = std::time::Instant::now();
                                    let gd = if let Some(dur) = budget_dur(goal_dump_timeout_s) {
                                        rt.block_on(plc::goal_dump_in_text_at(
                                            &repo_root,
                                            &file,
                                            &parent.text,
                                            dur,
                                            Some(s0.line),
                                            first_error_line_1,
                                        ))
                                        .ok()
                                    } else {
                                        bailed_total_timeout = true;
                                        record_event(
                                            "bailout_total_timeout",
                                            json!({ "where": "goal_dump_goal_first_smt" }),
                                        );
                                        None
                                    };
                                    let elapsed_ms = t0.elapsed().as_millis() as u64;
                                    last_probe_ms = Some(elapsed_ms);
                                    prof_goal_dump_ms =
                                        prof_goal_dump_ms.saturating_add(elapsed_ms);
                                    let pp = gd.as_ref().and_then(|v| v.get("pp_dump"));
                                    let state_key = pp
                                        .and_then(|pp| hash_state_key(pp))
                                        .unwrap_or(UNKNOWN_STATE_KEY);
                                    let n_goals = pp
                                        .and_then(|pp| pp.get("goals"))
                                        .and_then(|v| v.as_array())
                                        .map(|a| a.len())
                                        .unwrap_or(0);
                                    let hyps_total = pp
                                        .and_then(|pp| pp.get("goals"))
                                        .and_then(|v| v.as_array())
                                        .map(|a| {
                                            a.iter()
                                                .map(|g| {
                                                    g.get("hyps")
                                                        .and_then(|h| h.as_array())
                                                        .map(|x| x.len())
                                                        .unwrap_or(0)
                                                })
                                                .sum::<usize>()
                                        })
                                        .unwrap_or(0);
                                    let target = pp
                                        .and_then(|pp| pp.get("goals"))
                                        .and_then(|v| v.as_array())
                                        .and_then(|a| a.first())
                                        .and_then(|g| g.get("pretty"))
                                        .and_then(|v| v.as_str())
                                        .and_then(|s| {
                                            s.lines().find_map(|ln| {
                                                ln.trim_start()
                                                    .strip_prefix("⊢")
                                                    .map(|r| r.trim().to_string())
                                            })
                                        })
                                        .unwrap_or_default();
                                    let hyps_texts: Vec<String> = pp
                                        .and_then(|pp| pp.get("goals"))
                                        .and_then(|v| v.as_array())
                                        .and_then(|a| a.first())
                                        .and_then(|g| g.get("hyps"))
                                        .and_then(|v| v.as_array())
                                        .map(|a| {
                                            a.iter()
                                                .filter_map(|h| {
                                                    h.get("text")
                                                        .and_then(|v| v.as_str())
                                                        .map(|s| s.to_string())
                                                })
                                                .collect::<Vec<_>>()
                                        })
                                        .unwrap_or_default();
                                    let tup = (state_key, n_goals, hyps_total, target.clone());
                                    goal_dump_cache.insert(key, tup.clone());
                                    goal_dump_hyps_cache.insert(key, hyps_texts.clone());
                                    if let Some(cd) = cache_dir.as_ref() {
                                        cache_write_goal_dump(
                                            cd,
                                            th,
                                            parent.text.len(),
                                            s0.line,
                                            state_key,
                                            n_goals,
                                            hyps_total,
                                            &target,
                                            &hyps_texts,
                                        );
                                    }

                                    // SMT precheck (same logic as above, but after the no-cache path).
                                    if smt_precheck {
                                        if let Some(pp0) = pp {
                                            let goal_sig = hash_text(&target);
                                            let ck = (state_key, goal_sig, smt_depth);
                                            let mut entails_opt: Option<bool> = None;
                                            if let Some(v) = smt_entails_cache.get(&ck).copied() {
                                                smt_cache_hits += 1;
                                                entails_opt = Some(v);
                                            } else if let Some(cd2) = cache_dir.as_ref() {
                                                if let Some(v) = cache_read_smt_entails(
                                                    cd2, state_key, goal_sig, smt_depth,
                                                ) {
                                                    smt_cache_hits += 1;
                                                    smt_entails_cache.insert(ck, v);
                                                    entails_opt = Some(v);
                                                }
                                            }
                                            if entails_opt.is_none() {
                                                smt_cache_misses += 1;
                                                let t_smt0 = std::time::Instant::now();
                                                let (entails, attempts) =
                                                    smt_entails_from_pp_dump_escalating(
                                                        pp0,
                                                        smt_timeout_ms,
                                                        smt_seed,
                                                        smt_depth,
                                                        &smt_solver_norm,
                                                        smt_aggressive,
                                                        &mut smt_reuse,
                                                        &mut smt_entails_trace,
                                                    )
                                                    .unwrap_or_else(|e| {
                                                        smt_errors += 1;
                                                        smt_last_error =
                                                            Some(truncate_str(&e, 400));
                                                        (None, 1)
                                                    });
                                                smt_entails_attempts =
                                                    smt_entails_attempts.saturating_add(attempts);
                                                if attempts > 1 {
                                                    smt_entails_escalations =
                                                        smt_entails_escalations.saturating_add(
                                                            attempts.saturating_sub(1),
                                                        );
                                                }
                                                prof_smt_ms = prof_smt_ms.saturating_add(
                                                    t_smt0.elapsed().as_millis() as u64,
                                                );
                                                if let Some(b) = entails {
                                                    smt_entails_cache.insert(ck, b);
                                                    if let Some(cd2) = cache_dir.as_ref() {
                                                        cache_write_smt_entails(
                                                            cd2, state_key, goal_sig, smt_depth, b,
                                                        );
                                                    }
                                                    entails_opt = Some(b);
                                                }
                                            }
                                            if entails_opt == Some(true) {
                                                best = Some((-1, s0));
                                                break;
                                            }
                                        }
                                    }
                                    tup
                                };

                                // Difficulty heuristic: fewer goals, fewer hyps, smaller state_key==0 means unknown (penalize).
                                let unknown_pen = if state_key == UNKNOWN_STATE_KEY {
                                    10_000
                                } else {
                                    0
                                };
                                let meta_vars_pen = if goal_meta_penalty > 0 {
                                    // If the target contains metavariables, treat it as harder to avoid wasting budget.
                                    let th = hash_text(&parent.text);
                                    let k = (th, parent.text.len(), s0.line);
                                    let target = goal_dump_cache
                                        .get(&k)
                                        .map(|(_, _, _, t)| t.clone())
                                        .unwrap_or_default();
                                    let meta =
                                        target.matches("?m").count() + target.matches("?_").count();
                                    (meta as i64).saturating_mul(goal_meta_penalty)
                                } else {
                                    0
                                };
                                let score0 = unknown_pen as i64
                                    + (n_goals as i64 * 100)
                                    + hyps_total as i64
                                    + meta_vars_pen;

                                // Optional SMT hint: if the target is implied by (some) integer constraints in the local context,
                                // treat this hole as very easy (it likely succumbs to `omega`/`linarith`/`simp`).
                                let score = {
                                    let mut score = score0;
                                    if smt_precheck && state_key != UNKNOWN_STATE_KEY {
                                        let th = hash_text(&parent.text);
                                        let k = (th, parent.text.len(), s0.line);
                                        let target = goal_dump_cache
                                            .get(&k)
                                            .map(|(_, _, _, t)| t.clone())
                                            .unwrap_or_default();
                                        if !target.is_empty() {
                                            let goal_sig = hash_text(&target);
                                            let ck = (state_key, goal_sig, smt_depth);
                                            let mut cached = smt_entails_cache.get(&ck).copied();
                                            if cached.is_none() {
                                                if let Some(cd) = cache_dir.as_ref() {
                                                    if let Some(ent) = cache_read_smt_entails(
                                                        cd, state_key, goal_sig, smt_depth,
                                                    ) {
                                                        cached = Some(ent);
                                                        smt_entails_cache.insert(ck, ent);
                                                        smt_cache_hits += 1;
                                                    }
                                                }
                                            }
                                            if cached.is_none() {
                                                // No extra Lean calls. If we have cached hypothesis text for this hole,
                                                // we can still run SMT without Lean (best-effort LIA).
                                                smt_cache_misses += 1;
                                                let hyps_texts = if let Some(xs) =
                                                    goal_dump_hyps_cache.get(&k).cloned()
                                                {
                                                    goal_dump_hyps_cache_hits += 1;
                                                    xs
                                                } else if let Some(cd) = cache_dir.as_ref() {
                                                    if let Some(xs) =
                                                        cache_read_goal_dump_hyps_texts(
                                                            cd,
                                                            th,
                                                            parent.text.len(),
                                                            s0.line,
                                                        )
                                                    {
                                                        goal_dump_hyps_cache_hits += 1;
                                                        goal_dump_hyps_cache.insert(k, xs.clone());
                                                        xs
                                                    } else {
                                                        goal_dump_hyps_cache_misses += 1;
                                                        Vec::new()
                                                    }
                                                } else {
                                                    goal_dump_hyps_cache_misses += 1;
                                                    Vec::new()
                                                };
                                                if !hyps_texts.is_empty() {
                                                    let t0 = std::time::Instant::now();
                                                    let (ent, attempts) =
                                                        smt_entails_from_hyps_target_escalating(
                                                            &hyps_texts,
                                                            &target,
                                                            smt_timeout_ms,
                                                            smt_seed,
                                                            smt_depth,
                                                            &smt_solver_norm,
                                                            smt_aggressive,
                                                            &mut smt_reuse,
                                                            &mut smt_entails_trace,
                                                        )
                                                        .unwrap_or_else(|e| {
                                                            smt_errors += 1;
                                                            smt_last_error =
                                                                Some(truncate_str(&e, 400));
                                                            (None, 1)
                                                        });
                                                    smt_entails_attempts = smt_entails_attempts
                                                        .saturating_add(attempts);
                                                    if attempts > 1 {
                                                        smt_entails_escalations =
                                                            smt_entails_escalations.saturating_add(
                                                                attempts.saturating_sub(1),
                                                            );
                                                    }
                                                    prof_smt_ms = prof_smt_ms.saturating_add(
                                                        t0.elapsed().as_millis() as u64,
                                                    );
                                                    if let Some(ent) = ent {
                                                        smt_entails_cache.insert(ck, ent);
                                                        if let Some(cd) = cache_dir.as_ref() {
                                                            cache_write_smt_entails(
                                                                cd, state_key, goal_sig, smt_depth,
                                                                ent,
                                                            );
                                                        }
                                                        cached = Some(ent);
                                                    }
                                                }
                                            }
                                            if cached == Some(true) {
                                                score = score.saturating_sub(10_000);
                                            }
                                        }
                                    }
                                    score
                                };
                                if best.as_ref().map(|(b, _)| score < *b).unwrap_or(true) {
                                    best = Some((score, s0));
                                }
                                if let Some(ms) = last_probe_ms {
                                    if ms >= goal_first_slow_ms {
                                        break;
                                    }
                                }
                            }
                            best.map(|(_, s)| s)
                        }
                    } else if let Some(fl) = parent.focus_line.or(first_error_line_1) {
                        // Choose the `sorry` closest to `fl` to keep the search locally focused.
                        locs.iter()
                            .min_by_key(|l| (l.line as i64 - fl as i64).abs())
                            .cloned()
                    } else {
                        plc::select_primary_sorry(first_error_line_1, &locs)
                    };
                    let Some(sel) = selected else {
                        continue;
                    };
                    let region = (sel.region_start, sel.region_end);

                    // Heuristic: if the `sorry` is inside a `by` block, we should use tactic scripts
                    // (`simp`, `apply`, etc.) rather than `by\n ...`.
                    let is_tactic_context = {
                        let lines: Vec<&str> = parent.text.lines().collect();
                        let line_idx0 = sel.line.saturating_sub(1);
                        let indent = sel
                            .line_text
                            .chars()
                            .take_while(|c| *c == ' ' || *c == '\t')
                            .count();
                        // More robust than “first smaller indent line”:
                        // in Lean, the `:= by` opener can occur on a *more-indented* continuation line
                        // (e.g. multi-line lemma signatures). We therefore scan upward until we hit a
                        // declaration boundary, looking for tactic openers.
                        let mut found = false;
                        if line_idx0 < lines.len() {
                            let start = line_idx0.saturating_sub(80);
                            for k in (start..line_idx0).rev() {
                                let l = lines[k];
                                let t = l.trim();
                                if t.is_empty() || t.starts_with("--") {
                                    continue;
                                }
                                // Positive signals.
                                if t == "by"
                                    || t.contains(":= by")
                                    || t.contains("=> by")
                                    || t.starts_with('·')
                                    || t.starts_with("case ")
                                {
                                    found = true;
                                    break;
                                }
                                // Stop at a declaration boundary at smaller indentation.
                                let ind_k =
                                    l.chars().take_while(|c| *c == ' ' || *c == '\t').count();
                                if ind_k < indent
                                    && (t.starts_with("lemma ")
                                        || t.starts_with("theorem ")
                                        || t.starts_with("def ")
                                        || t.starts_with("instance ")
                                        || t.starts_with("structure "))
                                {
                                    break;
                                }
                            }
                        }
                        found
                    };

                    let parent_first_error = parent
                        .verify_summary
                        .as_ref()
                        .and_then(|s| s.get("first_error"))
                        .and_then(|v| v.as_str());

                    // For `lean-try`, prefer calling the Lean oracle at the *current* hole (bounded + cached),
                    // then fall back to deterministic fill candidates.
                    let mut oracle_candidates: Option<Vec<String>> = None;
                    let mut force_fill_candidates = false;
                    if candidates_mode == "lean-try"
                        && lean_oracle_per_node
                        && lean_oracle_calls < lean_oracle_max_calls
                    {
                        let focus_line = sel.line;
                        let h = hash_text(&parent.text);
                        let key = (h, parent.text.len(), focus_line);

                        // First: try state-key cache only if we already *have* a state key for this hole.
                        // If we don't, go straight to `lean_suggest_in_text_at` (it returns a pp_dump and
                        // we can seed the state cache from that), avoiding an extra Lean run.
                        if !force_fill_candidates {
                            if let Some((sk, _, _, _)) = goal_dump_cache.get(&key) {
                                goal_dump_cache_hits += 1;
                                if *sk != UNKNOWN_STATE_KEY {
                                    if let Some(xs) = lean_state_candidates_cache.get(sk) {
                                        lean_state_cache_hits += 1;
                                        oracle_candidates = Some(xs.clone());
                                    } else {
                                        lean_state_cache_misses += 1;
                                    }
                                }
                            }
                        }

                        if let Some(xs) = lean_oracle_cache.get(&key) {
                            lean_oracle_cache_hits += 1;
                            oracle_candidates = Some(xs.clone());
                        } else {
                            lean_oracle_cache_misses += 1;
                            let first_error_line_1 = parent
                                .verify_summary
                                .as_ref()
                                .and_then(|s| s.get("first_error_loc"))
                                .and_then(|l| l.get("line"))
                                .and_then(|v| v.as_u64())
                                .map(|x| x as usize);
                            let t0 = std::time::Instant::now();
                            record_event(
                                "oracle_call",
                                json!({
                                    "line": focus_line,
                                    "timeout_s": oracle_timeout_s,
                                    "cache": { "hit": false },
                                    "first_error_line": first_error_line_1,
                                }),
                            );
                            let Some(dur) = budget_dur(oracle_timeout_s) else {
                                bailed_total_timeout = true;
                                record_event(
                                    "bailout_total_timeout",
                                    json!({ "where": "oracle_call" }),
                                );
                                break;
                            };
                            let ls_res = rt.block_on(plc::lean_suggest_in_text_at(
                                &repo_root,
                                &file,
                                &parent.text,
                                dur,
                                Some(focus_line),
                                first_error_line_1,
                            ));
                            let ls_err = ls_res.as_ref().err().cloned();
                            let ls = ls_res.ok();
                            let oracle_ms = t0.elapsed().as_millis() as u64;
                            prof_lean_suggest_ms = prof_lean_suggest_ms.saturating_add(oracle_ms);
                            lean_oracle_calls += 1;
                            let has_pp_dump = ls.as_ref().and_then(|v| v.get("pp_dump")).is_some();
                            let oracle_retry_attempted = ls
                                .as_ref()
                                .and_then(|v| v.get("oracle_retry"))
                                .and_then(|v| v.get("attempted"))
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false);
                            let suggs = ls
                                .as_ref()
                                .and_then(|v| v.get("suggestions"))
                                .and_then(|v| v.as_array())
                                .map(|a| {
                                    a.iter()
                                        .filter_map(|x| x.as_str().map(|s| s.to_string()))
                                        .collect::<Vec<_>>()
                                })
                                .unwrap_or_default();
                            let verify_raw = ls
                                .as_ref()
                                .and_then(|v| v.get("verify"))
                                .and_then(|v| v.get("raw"));
                            let verify_ok = verify_raw
                                .and_then(|v| v.get("ok"))
                                .and_then(|v| v.as_bool());
                            let verify_timeout = verify_raw
                                .and_then(|v| v.get("timeout"))
                                .and_then(|v| v.as_bool());
                            let verify_cmd0 = verify_raw
                                .and_then(|v| v.get("cmd"))
                                .and_then(|v| v.as_array())
                                .and_then(|a| a.first())
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string());
                            let verify_stderr_first = verify_raw
                                .and_then(|v| v.get("stderr"))
                                .and_then(|v| v.as_str())
                                .and_then(|s| s.lines().find(|l| !l.trim().is_empty()))
                                .map(|s| s.trim().to_string());
                            let suggestions_preview: Option<Vec<String>> = if log_level >= 2 {
                                Some(
                                    suggs
                                        .iter()
                                        .take(3)
                                        .map(|s| {
                                            let t = s.trim().to_string();
                                            if t.chars().count() > 400 {
                                                let head: String = t.chars().take(400).collect();
                                                format!("{head}...")
                                            } else {
                                                t
                                            }
                                        })
                                        .collect(),
                                )
                            } else {
                                None
                            };
                            record_event(
                                "oracle_result",
                                json!({
                                    "ms": oracle_ms,
                                    "ok": ls.is_some(),
                                    "error": ls_err,
                                    "suggestions_n": suggs.len(),
                                    "suggestions_preview": suggestions_preview,
                                    "has_pp_dump": has_pp_dump,
                                    "oracle_retry_attempted": oracle_retry_attempted,
                                    "verify_ok": verify_ok,
                                    "verify_timeout": verify_timeout,
                                    "verify_cmd0": verify_cmd0,
                                    "verify_stderr_first": verify_stderr_first,
                                }),
                            );
                            // Also try to derive some candidates from the goal snapshot if available.
                            let mut derived: Vec<String> = Vec::new();
                            let mut goal_hash: Option<u64> = None;
                            if let Some(pp) = ls.as_ref().and_then(|v| v.get("pp_dump")) {
                                // Goal/state key (factorized).
                                goal_hash = hash_state_key(pp);
                                // Record into goal-dump cache so we can reuse the state key later (even across edits).
                                let n_goals = pp
                                    .get("goals")
                                    .and_then(|v| v.as_array())
                                    .map(|a| a.len())
                                    .unwrap_or(0);
                                let hyps_total = pp
                                    .get("goals")
                                    .and_then(|v| v.as_array())
                                    .map(|a| {
                                        a.iter()
                                            .map(|g| {
                                                g.get("hyps")
                                                    .and_then(|h| h.as_array())
                                                    .map(|x| x.len())
                                                    .unwrap_or(0)
                                            })
                                            .sum::<usize>()
                                    })
                                    .unwrap_or(0);
                                let target = pp
                                    .get("goals")
                                    .and_then(|v| v.as_array())
                                    .and_then(|a| a.first())
                                    .and_then(|g| g.get("pretty"))
                                    .and_then(|v| v.as_str())
                                    .and_then(|s| {
                                        s.lines().find_map(|ln| {
                                            ln.trim_start()
                                                .strip_prefix("⊢")
                                                .map(|r| r.trim().to_string())
                                        })
                                    })
                                    .unwrap_or_default();
                                let sk = goal_hash.unwrap_or(UNKNOWN_STATE_KEY);
                                goal_dump_cache
                                    .insert(key, (sk, n_goals, hyps_total, target.clone()));
                                // Best-effort: record local-context lines so SMT precheck can run without re-calling Lean.
                                let hyps_texts: Vec<String> = pp
                                    .get("goals")
                                    .and_then(|v| v.as_array())
                                    .and_then(|a| a.first())
                                    .and_then(|g| g.get("hyps"))
                                    .and_then(|v| v.as_array())
                                    .map(|a| {
                                        a.iter()
                                            .filter_map(|h| {
                                                h.get("text")
                                                    .and_then(|v| v.as_str())
                                                    .map(|s| s.to_string())
                                            })
                                            .collect::<Vec<_>>()
                                    })
                                    .unwrap_or_default();
                                goal_dump_hyps_cache.insert(key, hyps_texts.clone());
                                if let Some(cd) = cache_dir.as_ref() {
                                    cache_write_goal_dump(
                                        cd,
                                        h,
                                        parent.text.len(),
                                        focus_line,
                                        sk,
                                        n_goals,
                                        hyps_total,
                                        &target,
                                        &hyps_texts,
                                    );
                                }
                                // Use first goal pretty as a cheap heuristic source.
                                if let Some(pretty) = pp
                                    .get("goals")
                                    .and_then(|v| v.as_array())
                                    .and_then(|xs| xs.first())
                                    .and_then(|v| v.get("pretty"))
                                    .and_then(|v| v.as_str())
                                {
                                    derived =
                                        plc::derive_candidates_from_goal_pretty_with_hint_rules(
                                            pretty,
                                            &hint_rules,
                                        );
                                }
                                // Also derive from the target + hypothesis snippets (often includes type hints).
                                if !target.trim().is_empty() {
                                    let mut more =
                                        plc::derive_candidates_from_goal_context_with_hint_rules(
                                            &hyps_texts,
                                            &target,
                                            &hint_rules,
                                        );
                                    derived.append(&mut more);
                                }
                            }
                            if dedup_goal_expansions {
                                if let Some(gh) = goal_hash {
                                    if !expanded_goal_hashes.insert(gh) {
                                        // We've already expanded this goal state; skip oracle-driven expansions.
                                        lean_oracle_goal_dedup_skips += 1;
                                        // Still allow deterministic fill candidates later.
                                        force_fill_candidates = true;
                                    }
                                }
                            }

                            if !force_fill_candidates {
                                // Convert oracle suggestions into replacement candidates at this hole.
                                let mut xs: Vec<String> = Vec::new();
                                for s in suggs.into_iter().take(10) {
                                    let t = s.trim();
                                    if t.is_empty() {
                                        continue;
                                    }
                                    if t.contains("?_") {
                                        // Skeletonize into nested bullet goals.
                                        let holes = t.matches("?_").count().max(1).min(6);
                                        let mut out = String::new();
                                        out.push_str(t);
                                        for _ in 0..holes {
                                            out.push_str("\n·\n");
                                            out.push_str(&safe_tactic_fallback_block("  "));
                                        }
                                        xs.push(out);
                                    } else {
                                        // For tactic-level commands (especially `apply`/`refine`/`exact`), make them safe:
                                        // run the command, then attempt to close all goals, then fall back to `sorry`.
                                        //
                                        // This avoids hard error nodes at deeper depths.
                                        let mut out = String::new();
                                        out.push_str(t);
                                        out.push_str("\nall_goals\n");
                                        out.push_str(&safe_tactic_fallback_block("  "));
                                        xs.push(out);
                                    }
                                }
                                // Append derived candidates (cheap heuristics) as a fallback.
                                xs.extend(derived.into_iter());
                                xs = sanitize_candidates(xs);
                                if !xs.is_empty() {
                                    lean_oracle_cache.insert(key, xs.clone());
                                    oracle_candidates = Some(xs);
                                }
                            }
                        }
                    }

                    // If we successfully generated candidates and we know the goal state key, memoize by state key too.
                    if candidates_mode == "lean-try" {
                        let focus_line = sel.line;
                        let h = hash_text(&parent.text);
                        let key = (h, parent.text.len(), focus_line);
                        if let Some((sk, _, _, _)) = goal_dump_cache.get(&key).cloned() {
                            if sk != UNKNOWN_STATE_KEY {
                                if let Some(xs) = oracle_candidates.as_ref() {
                                    lean_state_candidates_cache
                                        .entry(sk)
                                        .or_insert_with(|| xs.clone());
                                }
                            }
                        }
                    }

                    let base_candidates = if !force_fill_candidates {
                        if let Some(xs) = oracle_candidates.as_ref() {
                            xs
                        } else if candidates_mode == "lean-try" && parent.depth > 0 {
                            &candidates_fill
                        } else {
                            &candidates_root
                        }
                    } else {
                        &candidates_fill
                    };
                    let candidates_here0 =
                        adapt_candidates_for_error(base_candidates, parent_first_error);
                    let mut candidates_here0 = adapt_candidates_for_sorry_context(
                        &candidates_here0,
                        &sel.line_text,
                        is_tactic_context,
                    );
                    // If the hole is inside an existing `by` block, prefer tactic-shaped candidates.
                    // (The default `det` candidate list is proof-term shaped: `by ...`.)
                    if is_tactic_context {
                        // Small goal-shape heuristics to make the top-of-beam candidates matter.
                        // (Beam search will only try the first few candidates per node.)
                        let (target_shape, hyps_shape) = {
                            let th = hash_text(&parent.text);
                            let k = (th, parent.text.len(), sel.line);
                            let mut target = goal_dump_cache
                                .get(&k)
                                .map(|(_, _, _, t)| t.clone())
                                .unwrap_or_default();
                            let mut hyps =
                                goal_dump_hyps_cache.get(&k).cloned().unwrap_or_default();
                            if hyps.is_empty() {
                                // Fallback: use the most recent goal dump snapshot (no extra Lean calls).
                                if let Some(gd) = goal_dump_v.as_ref() {
                                    if let Some(a) = gd
                                        .get("pp_dump")
                                        .and_then(|v| v.get("goals"))
                                        .and_then(|v| v.as_array())
                                    {
                                        if let Some(g0) = a.first() {
                                            if target.trim().is_empty() {
                                                if let Some(pretty) =
                                                    g0.get("pretty").and_then(|v| v.as_str())
                                                {
                                                    target = pretty
                                                        .lines()
                                                        .find_map(|ln| {
                                                            ln.trim_start()
                                                                .strip_prefix("⊢")
                                                                .map(|r| r.trim().to_string())
                                                        })
                                                        .unwrap_or_default();
                                                }
                                            }
                                            if let Some(hs) =
                                                g0.get("hyps").and_then(|v| v.as_array())
                                            {
                                                hyps = hs
                                                    .iter()
                                                    .filter_map(|h| {
                                                        h.get("text")
                                                            .and_then(|v| v.as_str())
                                                            .map(|s| s.to_string())
                                                    })
                                                    .collect();
                                            }
                                        }
                                    }
                                }
                            }
                            (target, hyps)
                        };
                        // Dynamic “structure” candidates inferred from local context.
                        let mut dynamic: Vec<String> = Vec::new();
                        let even_hyp = hyps_shape.iter().find_map(|h| {
                            let (lhs, rhs) = h.split_once(':')?;
                            if rhs.contains("Even") {
                                Some(lhs.trim().split_whitespace().next()?.to_string())
                            } else {
                                None
                            }
                        });
                        let mut div_hyps: Vec<String> = Vec::new();
                        for h in &hyps_shape {
                            if let Some((lhs, rhs)) = h.split_once(':') {
                                if rhs.contains('∣') {
                                    if let Some(nm) = lhs.trim().split_whitespace().next() {
                                        if !nm.is_empty() {
                                            div_hyps.push(nm.to_string());
                                        }
                                    }
                                }
                            }
                        }
                        div_hyps.sort();
                        div_hyps.dedup();
                        if target_shape.contains("∣") {
                            if let Some(hn) = even_hyp.as_ref() {
                                dynamic
                                    .push(format!("rcases {hn} with ⟨k, rfl⟩; exact ⟨k, by ring⟩"));
                                dynamic.push(format!(
                                    "rcases {hn} with ⟨k, rfl⟩; exact ⟨k, by omega⟩"
                                ));
                            }
                            // Very common shape: prove divisibility of a difference from two divisibility hypotheses.
                            // Example: `(4:ℤ) ∣ a` and `(4:ℤ) ∣ b` → `(4:ℤ) ∣ a - b`.
                            if div_hyps.len() >= 2 {
                                let ha = &div_hyps[0];
                                let hb = &div_hyps[1];
                                dynamic.push(format!(
                                    "rcases {ha} with ⟨ka, rfl⟩; rcases {hb} with ⟨kb, rfl⟩; exact ⟨ka - kb, by ring⟩"
                                ));
                            }
                        }

                        let prioritize_even_div = target_shape.contains("∣") && even_hyp.is_some();
                        let mut pref: Vec<String> = vec![
                            // Algebra/arithmetic workhorses.
                            "ring".to_string(),
                            "ring_nf".to_string(),
                            "omega".to_string(),
                            "linarith".to_string(),
                            "nlinarith".to_string(),
                            "norm_cast".to_string(),
                            "norm_num".to_string(),
                            "simp".to_string(),
                            "simp_all".to_string(),
                            "aesop".to_string(),
                            // Mathlib suggestions can be surprisingly effective for “obvious” goals.
                            "exact?".to_string(),
                            "apply?".to_string(),
                        ];
                        // Add a small, bounded set of goal-derived candidates.
                        // This is repo-agnostic: it only uses the pretty-printed goal + hypotheses surface.
                        //
                        // Important: this is a tactic hole, so candidates must be tactic-shaped.
                        // `derive_candidates_from_goal_context` returns term-shaped `by ...` blocks;
                        // convert them into tactic scripts.
                        let mut derived = plc::derive_candidates_from_goal_context_with_hint_rules(
                            &hyps_shape,
                            &target_shape,
                            &hint_rules,
                        );
                        derived.truncate(12);
                        derived = plc::tree_search::adapt_candidates_for_tactic_hole(&derived);
                        // Keep dynamic candidates at the very front (they're goal-specific and
                        // should land in the beam). Derived candidates come next; then pref list.
                        if !dynamic.is_empty() {
                            let mut v = dynamic;
                            // Keep derived close to the front, but after the dynamic candidates.
                            v.append(&mut derived);
                            v.append(&mut pref);
                            pref = v;
                        } else if !derived.is_empty() {
                            derived.append(&mut pref);
                            pref = derived;
                        }
                        if !prioritize_even_div {
                            // For non-divisibility goals, prefer simp/aesop first.
                            pref.rotate_left(10);
                        }
                        pref.append(&mut candidates_here0);
                        candidates_here0 = sanitize_candidates(pref);
                    }

                    // Optional: if deterministic tactics stalled, opportunistically ask the LLM
                    // for more candidates for this exact region.
                    let mut candidates_here = if escalate_llm
                        && candidates_mode != "llm"
                        && is_made_no_progress(parent_first_error)
                    {
                        let payload = plc::build_region_patch_prompt(
                            &repo_root,
                            &file,
                            region.0,
                            region.1,
                            parent_first_error,
                        )?;
                        let mut system = payload.system.clone();
                        system.push_str("\n\nReturn a JSON array of 6 distinct candidate Lean replacements (strings). Each element must be a proof term only (no markdown fences).");
                        if !allow_sorry_candidates {
                            system.push_str("\n\nConstraints:\n- Do not use `sorry` or `admit` anywhere.\n- Return complete proof terms only (no placeholders).");
                        }
                        if let Some(gd) = goal_dump_v.as_ref() {
                            if let Some(pretty) = gd
                                .get("pp_dump")
                                .and_then(|v| v.get("goals"))
                                .and_then(|v| v.as_array())
                                .and_then(|xs| xs.first())
                                .and_then(|v| v.get("pretty"))
                                .and_then(|v| v.as_str())
                            {
                                system.push_str("\n\nGoal snapshot (pretty):\n");
                                system.push_str(pretty);
                            }
                        }
                        // Also try to harvest a goal snapshot from `aesop`-style failures even when goal_dump is off.
                        if let Some(raw) = parent.verify_raw.as_ref() {
                            if let Some(stdout) = raw.get("stdout").and_then(|v| v.as_str()) {
                                if let Some(block) = extract_initial_goal_block(stdout) {
                                    system.push_str("\n\nLean log goal snapshot:\n");
                                    system.push_str(&block);
                                }
                            }
                        }
                        if let Some(notes) = research_notes_text.as_ref() {
                            system.push_str("\n\nResearch context (may be incomplete):\n");
                            system.push_str(notes);
                        }
                        let extra = rt.block_on(plc::llm::chat_completion(
                            &system,
                            &payload.user,
                            StdDuration::from_secs(llm_timeout_s),
                        ));
                        llm_escalate_attempts += 1;
                        if let Ok(res) = extra {
                            if let Some(xs) = parse_json_string_array(&res.content) {
                                llm_escalate_successes += 1;
                                let mut merged = candidates_here0.clone();
                                merged.extend(xs);
                                sanitize_candidates(merged)
                            } else {
                                llm_escalate_last_error =
                                    Some("llm_response_not_json_string_array".to_string());
                                candidates_here0
                            }
                        } else {
                            llm_escalate_last_error = Some(format!("{:?}", extra.err()));
                            candidates_here0
                        }
                    } else {
                        candidates_here0
                    };

                    let cap = effective_max_candidates_per_node;
                    let verify_k = effective_verify_k;

                    // Best-effort goal state key for this hole (if we have it).
                    let state_key_opt = {
                        let th = hash_text(&parent.text);
                        let k = (th, parent.text.len(), sel.line);
                        goal_dump_cache
                            .get(&k)
                            .map(|(sk, _, _, _)| *sk)
                            .filter(|sk| *sk != UNKNOWN_STATE_KEY)
                    };

                    // Optional SMT entailment signal for this hole (cache-only; no Lean calls here).
                    // This is used to re-rank candidates: if the goal is implied by a linear arithmetic fragment,
                    // prioritize arithmetic tactics; if not implied, de-prioritize them.
                    let (smt_entails_opt, mut smt_hint_json): (
                        Option<bool>,
                        Option<serde_json::Value>,
                    ) = if smt_precheck {
                        if let Some(sk) = state_key_opt {
                            let th = hash_text(&parent.text);
                            let k = (th, parent.text.len(), sel.line);
                            let target = goal_dump_cache
                                .get(&k)
                                .map(|(_, _, _, t)| t.clone())
                                .unwrap_or_default();
                            if target.is_empty() {
                                (
                                    None,
                                    Some(json!({
                                        "entails": serde_json::Value::Null,
                                        "source": "no_target",
                                        "state_key": sk,
                                        "goal_sig": serde_json::Value::Null,
                                    })),
                                )
                            } else if let Some(cd) = cache_dir.as_ref() {
                                let goal_sig = hash_text(&target);
                                let ck = (sk, goal_sig, smt_depth);
                                // If the user asked for SMT artifacts (proof/core/dumps), we still want to
                                // *use* the cached entailment result for ranking, but we should not skip
                                // producing artifacts just because we hit the cache.
                                let want_smt_artifacts =
                                    smt_dump || smt_unsat_core || smt_support || smt_proof;

                                if let Some(v) = smt_entails_cache.get(&ck).copied() {
                                    smt_cache_hits += 1;
                                    let mut hint = json!({
                                        "entails": v,
                                        "source": "mem",
                                        "state_key": sk,
                                        "goal_sig": goal_sig,
                                    });
                                    if want_smt_artifacts && v && !smt_artifacts_done.contains(&ck)
                                    {
                                        let hyps_texts = if let Some(xs) =
                                            goal_dump_hyps_cache.get(&k).cloned()
                                        {
                                            goal_dump_hyps_cache_hits += 1;
                                            xs
                                        } else if let Some(xs) = cache_read_goal_dump_hyps_texts(
                                            cd,
                                            th,
                                            parent.text.len(),
                                            sel.line,
                                        ) {
                                            goal_dump_hyps_cache_hits += 1;
                                            goal_dump_hyps_cache.insert(k, xs.clone());
                                            xs
                                        } else {
                                            goal_dump_hyps_cache_misses += 1;
                                            Vec::new()
                                        };
                                        if !hyps_texts.is_empty() {
                                            if smt_dump && smt_dumps_written < smt_dump_max as u64 {
                                                let pp_dump = json!({
                                                    "goals": [{
                                                        "pretty": format!("{}\n⊢ {}", hyps_texts.join("\n"), target),
                                                        "hyps": hyps_texts.iter().take(48).map(|s| json!({"text": s})).collect::<Vec<_>>()
                                                    }]
                                                });
                                                if let Some(script) =
                                                    plc::smt_lia::smt2_script_from_pp_dump(
                                                        &pp_dump,
                                                        smt_timeout_ms,
                                                        smt_seed,
                                                        smt_depth,
                                                    )
                                                {
                                                    let root = smt_dump_root(
                                                        &repo_root,
                                                        &smt_dump_dir_opt,
                                                    );
                                                    let path = maybe_write_smt2_dump(
                                                        &root, sk, goal_sig, smt_depth, &script,
                                                    );
                                                    smt_dumps_written += 1;
                                                    smt_dump_paths.push(path.clone());
                                                    smt_dump_last_path = Some(path.clone());
                                                    smt_dump_last_chars =
                                                        Some(script.chars().count());
                                                    smt_dump_last_preview =
                                                        Some(truncate_str(&script, 4_000));
                                                    if let Some(obj) = hint.as_object_mut() {
                                                        obj.insert(
                                                            "smt2_dump".to_string(),
                                                            json!(path),
                                                        );
                                                        obj.insert(
                                                            "smt2_dump_chars".to_string(),
                                                            json!(script.chars().count()),
                                                        );
                                                        obj.insert(
                                                            "smt2_dump_preview".to_string(),
                                                            json!(truncate_str(&script, 2_000)),
                                                        );
                                                    }
                                                }
                                            }
                                            if (smt_unsat_core || smt_support || smt_proof) && v {
                                                // Rebuild pp_dump in the shape required by the helper fns.
                                                let pp_dump = json!({
                                                    "goals": [{
                                                        "pretty": format!("{}\n⊢ {}", hyps_texts.join("\n"), target),
                                                        "hyps": hyps_texts.iter().take(48).map(|s| json!({"text": s})).collect::<Vec<_>>()
                                                    }]
                                                });
                                                if let Some(obj) = hint.as_object_mut() {
                                                    if smt_unsat_core || smt_support {
                                                        let core =
                                                            plc::smt_lia::unsat_core_from_pp_dump(
                                                                &pp_dump,
                                                                smt_timeout_ms,
                                                                smt_seed,
                                                                smt_depth,
                                                                if smt_unsat_core {
                                                                    smt_unsat_core_max
                                                                } else {
                                                                    smt_support_max
                                                                },
                                                            )
                                                            .unwrap_or(None)
                                                            .unwrap_or(serde_json::Value::Null);
                                                        obj.insert("unsat_core".to_string(), core);
                                                    }
                                                    if smt_proof
                                                        && smt_proof_done
                                                            .insert((sk, goal_sig, smt_depth))
                                                    {
                                                        smt_proof_attempts =
                                                            smt_proof_attempts.saturating_add(1);
                                                        match plc::smt_lia::unsat_proof_from_pp_dump(
                                                            &pp_dump,
                                                            smt_timeout_ms,
                                                            smt_seed,
                                                            smt_depth,
                                                            smt_proof_max_chars,
                                                        ) {
                                                            Ok(Some(pf)) => {
                                                                smt_proofs_captured =
                                                                    smt_proofs_captured
                                                                        .saturating_add(1);
                                                                smt_proof_last = Some(pf.clone());
                                                                obj.insert(
                                                                    "unsat_proof".to_string(),
                                                                    pf,
                                                                );
                                                            }
                                                            Ok(None) => {
                                                                obj.insert(
                                                                    "unsat_proof".to_string(),
                                                                    serde_json::Value::Null,
                                                                );
                                                                smt_proof_last_error = Some(
                                                                    "proof_unavailable".to_string(),
                                                                );
                                                            }
                                                            Err(e) => {
                                                                obj.insert(
                                                                    "unsat_proof".to_string(),
                                                                    serde_json::Value::Null,
                                                                );
                                                                smt_proof_last_error =
                                                                    Some(truncate_str(&e, 400));
                                                            }
                                                        }
                                                    }
                                                    if smt_proof
                                                        && smt_proof_dump
                                                        && smt_proof_dump_done
                                                            .insert((sk, goal_sig, smt_depth))
                                                    {
                                                        smt_proof_dump_attempts =
                                                            smt_proof_dump_attempts
                                                                .saturating_add(1);
                                                        if let Ok(Some(pf_full)) =
                                                            plc::smt_lia::unsat_proof_from_pp_dump(
                                                                &pp_dump,
                                                                smt_timeout_ms,
                                                                smt_seed,
                                                                smt_depth,
                                                                smt_proof_dump_max_chars,
                                                            )
                                                        {
                                                            if let Some(preview) = pf_full
                                                                .get("preview")
                                                                .and_then(|v| v.as_str())
                                                            {
                                                                let total_chars = pf_full
                                                                    .get("chars")
                                                                    .and_then(|v| v.as_u64())
                                                                    .unwrap_or(0)
                                                                    as usize;
                                                                let preview_chars =
                                                                    preview.chars().count();
                                                                if total_chars
                                                                    <= smt_proof_dump_max_chars
                                                                    && preview_chars == total_chars
                                                                {
                                                                    let root = smt_proof_dump_root(
                                                                        &repo_root,
                                                                        &smt_proof_dump_dir_opt,
                                                                    );
                                                                    let path =
                                                                        maybe_write_smt_proof_dump(
                                                                            &root, sk, goal_sig,
                                                                            smt_depth, preview,
                                                                        );
                                                                    if !smt_proof_dump_paths
                                                                        .contains(&path)
                                                                    {
                                                                        smt_proof_dump_paths
                                                                            .push(path.clone());
                                                                        smt_proof_dump_written =
                                                                            smt_proof_dump_written
                                                                                .saturating_add(1);
                                                                    }
                                                                    obj.insert(
                                                                        "unsat_proof_path"
                                                                            .to_string(),
                                                                        json!(path),
                                                                    );
                                                                } else {
                                                                    smt_proof_dump_skipped_too_large =
                                                                        smt_proof_dump_skipped_too_large
                                                                            .saturating_add(1);
                                                                    smt_proof_dump_last_error = Some(
                                                                        format!(
                                                                            "proof_too_large_for_dump chars={total_chars} max_chars={}",
                                                                            smt_proof_dump_max_chars
                                                                        ),
                                                                    );
                                                                }
                                                            }
                                                        } else {
                                                            smt_proof_dump_last_error = Some(
                                                                "proof_unavailable".to_string(),
                                                            );
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    // Only mark artifacts as done if we actually attempted to build them.
                                    if want_smt_artifacts && v && !smt_artifacts_done.contains(&ck)
                                    {
                                        smt_artifacts_done.insert(ck);
                                    }

                                    (Some(v), Some(hint))
                                } else if let Some(v) =
                                    cache_read_smt_entails(cd, sk, goal_sig, smt_depth)
                                {
                                    smt_cache_hits += 1;
                                    smt_entails_cache.insert(ck, v);
                                    let mut hint = json!({
                                        "entails": v,
                                        "source": "disk",
                                        "state_key": sk,
                                        "goal_sig": goal_sig,
                                    });
                                    if want_smt_artifacts && v && !smt_artifacts_done.contains(&ck)
                                    {
                                        let hyps_texts = if let Some(xs) =
                                            goal_dump_hyps_cache.get(&k).cloned()
                                        {
                                            goal_dump_hyps_cache_hits += 1;
                                            xs
                                        } else if let Some(xs) = cache_read_goal_dump_hyps_texts(
                                            cd,
                                            th,
                                            parent.text.len(),
                                            sel.line,
                                        ) {
                                            goal_dump_hyps_cache_hits += 1;
                                            goal_dump_hyps_cache.insert(k, xs.clone());
                                            xs
                                        } else {
                                            goal_dump_hyps_cache_misses += 1;
                                            Vec::new()
                                        };
                                        if !hyps_texts.is_empty() {
                                            if smt_dump && smt_dumps_written < smt_dump_max as u64 {
                                                let pp_dump = json!({
                                                    "goals": [{
                                                        "pretty": format!("{}\n⊢ {}", hyps_texts.join("\n"), target),
                                                        "hyps": hyps_texts.iter().take(48).map(|s| json!({"text": s})).collect::<Vec<_>>()
                                                    }]
                                                });
                                                if let Some(script) =
                                                    plc::smt_lia::smt2_script_from_pp_dump(
                                                        &pp_dump,
                                                        smt_timeout_ms,
                                                        smt_seed,
                                                        smt_depth,
                                                    )
                                                {
                                                    let root = smt_dump_root(
                                                        &repo_root,
                                                        &smt_dump_dir_opt,
                                                    );
                                                    let path = maybe_write_smt2_dump(
                                                        &root, sk, goal_sig, smt_depth, &script,
                                                    );
                                                    smt_dumps_written += 1;
                                                    smt_dump_paths.push(path.clone());
                                                    smt_dump_last_path = Some(path.clone());
                                                    smt_dump_last_chars =
                                                        Some(script.chars().count());
                                                    smt_dump_last_preview =
                                                        Some(truncate_str(&script, 4_000));
                                                    if let Some(obj) = hint.as_object_mut() {
                                                        obj.insert(
                                                            "smt2_dump".to_string(),
                                                            json!(path),
                                                        );
                                                        obj.insert(
                                                            "smt2_dump_chars".to_string(),
                                                            json!(script.chars().count()),
                                                        );
                                                        obj.insert(
                                                            "smt2_dump_preview".to_string(),
                                                            json!(truncate_str(&script, 2_000)),
                                                        );
                                                    }
                                                }
                                            }
                                            if (smt_unsat_core || smt_support || smt_proof) && v {
                                                let pp_dump = json!({
                                                    "goals": [{
                                                        "pretty": format!("{}\n⊢ {}", hyps_texts.join("\n"), target),
                                                        "hyps": hyps_texts.iter().take(48).map(|s| json!({"text": s})).collect::<Vec<_>>()
                                                    }]
                                                });
                                                if let Some(obj) = hint.as_object_mut() {
                                                    if smt_unsat_core || smt_support {
                                                        let core =
                                                            plc::smt_lia::unsat_core_from_pp_dump(
                                                                &pp_dump,
                                                                smt_timeout_ms,
                                                                smt_seed,
                                                                smt_depth,
                                                                if smt_unsat_core {
                                                                    smt_unsat_core_max
                                                                } else {
                                                                    smt_support_max
                                                                },
                                                            )
                                                            .unwrap_or(None)
                                                            .unwrap_or(serde_json::Value::Null);
                                                        obj.insert("unsat_core".to_string(), core);
                                                    }
                                                    if smt_proof
                                                        && smt_proof_done
                                                            .insert((sk, goal_sig, smt_depth))
                                                    {
                                                        smt_proof_attempts =
                                                            smt_proof_attempts.saturating_add(1);
                                                        match plc::smt_lia::unsat_proof_from_pp_dump(
                                                            &pp_dump,
                                                            smt_timeout_ms,
                                                            smt_seed,
                                                            smt_depth,
                                                            smt_proof_max_chars,
                                                        ) {
                                                            Ok(Some(pf)) => {
                                                                smt_proofs_captured =
                                                                    smt_proofs_captured
                                                                        .saturating_add(1);
                                                                smt_proof_last = Some(pf.clone());
                                                                obj.insert(
                                                                    "unsat_proof".to_string(),
                                                                    pf,
                                                                );
                                                            }
                                                            Ok(None) => {
                                                                obj.insert(
                                                                    "unsat_proof".to_string(),
                                                                    serde_json::Value::Null,
                                                                );
                                                                smt_proof_last_error = Some(
                                                                    "proof_unavailable".to_string(),
                                                                );
                                                            }
                                                            Err(e) => {
                                                                obj.insert(
                                                                    "unsat_proof".to_string(),
                                                                    serde_json::Value::Null,
                                                                );
                                                                smt_proof_last_error =
                                                                    Some(truncate_str(&e, 400));
                                                            }
                                                        }
                                                    }
                                                    if smt_proof
                                                        && smt_proof_dump
                                                        && smt_proof_dump_done
                                                            .insert((sk, goal_sig, smt_depth))
                                                    {
                                                        smt_proof_dump_attempts =
                                                            smt_proof_dump_attempts
                                                                .saturating_add(1);
                                                        if let Ok(Some(pf_full)) =
                                                            plc::smt_lia::unsat_proof_from_pp_dump(
                                                                &pp_dump,
                                                                smt_timeout_ms,
                                                                smt_seed,
                                                                smt_depth,
                                                                smt_proof_dump_max_chars,
                                                            )
                                                        {
                                                            if let Some(preview) = pf_full
                                                                .get("preview")
                                                                .and_then(|v| v.as_str())
                                                            {
                                                                let total_chars = pf_full
                                                                    .get("chars")
                                                                    .and_then(|v| v.as_u64())
                                                                    .unwrap_or(0)
                                                                    as usize;
                                                                let preview_chars =
                                                                    preview.chars().count();
                                                                if total_chars
                                                                    <= smt_proof_dump_max_chars
                                                                    && preview_chars == total_chars
                                                                {
                                                                    let root = smt_proof_dump_root(
                                                                        &repo_root,
                                                                        &smt_proof_dump_dir_opt,
                                                                    );
                                                                    let path =
                                                                        maybe_write_smt_proof_dump(
                                                                            &root, sk, goal_sig,
                                                                            smt_depth, preview,
                                                                        );
                                                                    if !smt_proof_dump_paths
                                                                        .contains(&path)
                                                                    {
                                                                        smt_proof_dump_paths
                                                                            .push(path.clone());
                                                                        smt_proof_dump_written =
                                                                            smt_proof_dump_written
                                                                                .saturating_add(1);
                                                                    }
                                                                    obj.insert(
                                                                        "unsat_proof_path"
                                                                            .to_string(),
                                                                        json!(path),
                                                                    );
                                                                } else {
                                                                    smt_proof_dump_skipped_too_large =
                                                                        smt_proof_dump_skipped_too_large
                                                                            .saturating_add(1);
                                                                    smt_proof_dump_last_error = Some(
                                                                        format!(
                                                                            "proof_too_large_for_dump chars={total_chars} max_chars={}",
                                                                            smt_proof_dump_max_chars
                                                                        ),
                                                                    );
                                                                }
                                                            }
                                                        } else {
                                                            smt_proof_dump_last_error = Some(
                                                                "proof_unavailable".to_string(),
                                                            );
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    // Only mark artifacts as done if we actually attempted to build them.
                                    if want_smt_artifacts && v && !smt_artifacts_done.contains(&ck)
                                    {
                                        smt_artifacts_done.insert(ck);
                                    }

                                    (Some(v), Some(hint))
                                } else {
                                    // Try to compute from cached hyps_texts + target (no Lean).
                                    smt_cache_misses += 1;
                                    let hyps_texts =
                                        if let Some(xs) = goal_dump_hyps_cache.get(&k).cloned() {
                                            goal_dump_hyps_cache_hits += 1;
                                            xs
                                        } else if let Some(xs) = cache_read_goal_dump_hyps_texts(
                                            cd,
                                            th,
                                            parent.text.len(),
                                            sel.line,
                                        ) {
                                            goal_dump_hyps_cache_hits += 1;
                                            goal_dump_hyps_cache.insert(k, xs.clone());
                                            xs
                                        } else {
                                            goal_dump_hyps_cache_misses += 1;
                                            Vec::new()
                                        };
                                    if hyps_texts.is_empty() {
                                        (
                                            None,
                                            Some(json!({
                                                "entails": serde_json::Value::Null,
                                                "source": "no_hyps",
                                                "state_key": sk,
                                                "goal_sig": goal_sig,
                                            })),
                                        )
                                    } else {
                                        let t0 = std::time::Instant::now();
                                        let (ent, attempts) =
                                            smt_entails_from_hyps_target_escalating(
                                                &hyps_texts,
                                                &target,
                                                smt_timeout_ms,
                                                smt_seed,
                                                smt_depth,
                                                &smt_solver_norm,
                                                smt_aggressive,
                                                &mut smt_reuse,
                                                &mut smt_entails_trace,
                                            )
                                            .unwrap_or_else(|e| {
                                                smt_errors += 1;
                                                smt_last_error = Some(truncate_str(&e, 400));
                                                (None, 1)
                                            });
                                        smt_entails_attempts =
                                            smt_entails_attempts.saturating_add(attempts);
                                        if attempts > 1 {
                                            smt_entails_escalations = smt_entails_escalations
                                                .saturating_add(attempts.saturating_sub(1));
                                        }
                                        prof_smt_ms = prof_smt_ms
                                            .saturating_add(t0.elapsed().as_millis() as u64);
                                        if let Some(ent) = ent {
                                            smt_entails_cache.insert(ck, ent);
                                            cache_write_smt_entails(
                                                cd, sk, goal_sig, smt_depth, ent,
                                            );
                                            let mut hint = json!({
                                                "entails": ent,
                                                "source": "computed",
                                                "state_key": sk,
                                                "goal_sig": goal_sig,
                                            });
                                            if smt_dump && smt_dumps_written < smt_dump_max as u64 {
                                                let pp_dump = json!({
                                                    "goals": [{
                                                        "pretty": format!("{}\n⊢ {}", hyps_texts.join("\n"), target),
                                                        "hyps": hyps_texts.iter().take(48).map(|s| json!({"text": s})).collect::<Vec<_>>()
                                                    }]
                                                });
                                                if let Some(script) =
                                                    plc::smt_lia::smt2_script_from_pp_dump(
                                                        &pp_dump,
                                                        smt_timeout_ms,
                                                        smt_seed,
                                                        smt_depth,
                                                    )
                                                {
                                                    let root = smt_dump_root(
                                                        &repo_root,
                                                        &smt_dump_dir_opt,
                                                    );
                                                    let path = maybe_write_smt2_dump(
                                                        &root, sk, goal_sig, smt_depth, &script,
                                                    );
                                                    smt_dumps_written += 1;
                                                    smt_dump_paths.push(path.clone());
                                                    smt_dump_last_path = Some(path.clone());
                                                    smt_dump_last_chars =
                                                        Some(script.chars().count());
                                                    smt_dump_last_preview =
                                                        Some(truncate_str(&script, 4_000));
                                                    if let Some(obj) = hint.as_object_mut() {
                                                        obj.insert(
                                                            "smt2_dump".to_string(),
                                                            json!(path),
                                                        );
                                                        obj.insert(
                                                            "smt2_dump_chars".to_string(),
                                                            json!(script.chars().count()),
                                                        );
                                                        obj.insert(
                                                            "smt2_dump_preview".to_string(),
                                                            json!(truncate_str(&script, 2_000)),
                                                        );
                                                    }
                                                }
                                            }
                                            if (smt_unsat_core || smt_support || smt_proof) && ent {
                                                // Best-effort: rebuild a minimal pp_dump shape.
                                                // Used for: unsat cores, support hints, and (optional) UNSAT proof objects.
                                                let mut pretty = String::new();
                                                for h in hyps_texts.iter().take(48) {
                                                    pretty.push_str(h);
                                                    pretty.push('\n');
                                                }
                                                pretty.push_str("⊢ ");
                                                pretty.push_str(&target);
                                                let hyps_json: Vec<serde_json::Value> = hyps_texts
                                                    .iter()
                                                    .take(48)
                                                    .map(|s| json!({ "text": s }))
                                                    .collect();
                                                let pp_dump = json!({
                                                    "goals": [{
                                                        "pretty": pretty,
                                                        "hyps": hyps_json
                                                    }]
                                                });
                                                if let Some(obj) = hint.as_object_mut() {
                                                    if smt_unsat_core || smt_support {
                                                        let core =
                                                            plc::smt_lia::unsat_core_from_pp_dump(
                                                                &pp_dump,
                                                                smt_timeout_ms,
                                                                smt_seed,
                                                                smt_depth,
                                                                if smt_unsat_core {
                                                                    smt_unsat_core_max
                                                                } else {
                                                                    smt_support_max
                                                                },
                                                            )
                                                            .unwrap_or(None)
                                                            .unwrap_or(serde_json::Value::Null);
                                                        obj.insert("unsat_core".to_string(), core);
                                                    }
                                                    if smt_proof {
                                                        smt_proof_attempts =
                                                            smt_proof_attempts.saturating_add(1);
                                                        match plc::smt_lia::unsat_proof_from_pp_dump(
                                                            &pp_dump,
                                                            smt_timeout_ms,
                                                            smt_seed,
                                                            smt_depth,
                                                            smt_proof_max_chars,
                                                        ) {
                                                            Ok(Some(pf)) => {
                                                                smt_proofs_captured =
                                                                    smt_proofs_captured
                                                                        .saturating_add(1);
                                                                smt_proof_last = Some(pf.clone());
                                                                obj.insert(
                                                                    "unsat_proof".to_string(),
                                                                    pf,
                                                                );
                                                            }
                                                            Ok(None) => {
                                                                obj.insert(
                                                                    "unsat_proof".to_string(),
                                                                    serde_json::Value::Null,
                                                                );
                                                                smt_proof_last_error = Some(
                                                                    "proof_unavailable".to_string(),
                                                                );
                                                            }
                                                            Err(e) => {
                                                                obj.insert(
                                                                    "unsat_proof".to_string(),
                                                                    serde_json::Value::Null,
                                                                );
                                                                smt_proof_last_error =
                                                                    Some(truncate_str(&e, 400));
                                                            }
                                                        }
                                                    }
                                                    // Optional: write the proof object to disk for reproducible debugging.
                                                    //
                                                    // Note: this is best-effort and bounded by `--smt-proof-dump-max-chars`.
                                                    if smt_proof && smt_proof_dump {
                                                        smt_proof_dump_attempts =
                                                            smt_proof_dump_attempts
                                                                .saturating_add(1);
                                                        if let Ok(Some(pf_full)) =
                                                            plc::smt_lia::unsat_proof_from_pp_dump(
                                                                &pp_dump,
                                                                smt_timeout_ms,
                                                                smt_seed,
                                                                smt_depth,
                                                                smt_proof_dump_max_chars,
                                                            )
                                                        {
                                                            if let Some(preview) = pf_full
                                                                .get("preview")
                                                                .and_then(|v| v.as_str())
                                                            {
                                                                let total_chars = pf_full
                                                                    .get("chars")
                                                                    .and_then(|v| v.as_u64())
                                                                    .unwrap_or(0)
                                                                    as usize;
                                                                let preview_chars =
                                                                    preview.chars().count();

                                                                // IMPORTANT: only write a `.sexp` dump when we have the full proof text.
                                                                // If we write a truncated prefix, it is very likely to be syntactically invalid.
                                                                if total_chars
                                                                    <= smt_proof_dump_max_chars
                                                                    && preview_chars == total_chars
                                                                {
                                                                    let root = smt_proof_dump_root(
                                                                        &repo_root,
                                                                        &smt_proof_dump_dir_opt,
                                                                    );
                                                                    let path =
                                                                        maybe_write_smt_proof_dump(
                                                                            &root, sk, goal_sig,
                                                                            smt_depth, preview,
                                                                        );
                                                                    if !smt_proof_dump_paths
                                                                        .contains(&path)
                                                                    {
                                                                        smt_proof_dump_paths
                                                                            .push(path.clone());
                                                                        smt_proof_dump_written =
                                                                            smt_proof_dump_written
                                                                                .saturating_add(1);
                                                                    }
                                                                    obj.insert(
                                                                        "unsat_proof_path"
                                                                            .to_string(),
                                                                        json!(path),
                                                                    );
                                                                } else {
                                                                    smt_proof_dump_skipped_too_large =
                                                                        smt_proof_dump_skipped_too_large
                                                                            .saturating_add(1);
                                                                    smt_proof_dump_last_error = Some(
                                                                        format!(
                                                                            "proof_too_large_for_dump chars={total_chars} max_chars={}",
                                                                            smt_proof_dump_max_chars
                                                                        ),
                                                                    );
                                                                }
                                                            }
                                                        } else {
                                                            smt_proof_dump_last_error = Some(
                                                                "proof_unavailable".to_string(),
                                                            );
                                                        }
                                                    }
                                                    if let Some(v) = obj.get("unsat_core") {
                                                        let ctx_hyp_names: std::collections::HashSet<
                                                            String,
                                                        > = hyps_texts
                                                            .iter()
                                                            .filter_map(|s| hyp_name_from_text_line(s))
                                                            .collect();
                                                        if let Some(hs) = smt_support_lean_hints(
                                                            v,
                                                            &ctx_hyp_names,
                                                        ) {
                                                            obj.insert(
                                                                "smt_support_lean".to_string(),
                                                                json!(hs),
                                                            );
                                                        }
                                                    }
                                                }
                                            }
                                            (Some(ent), Some(hint))
                                        } else {
                                            if smt_dump && smt_dumps_written < smt_dump_max as u64 {
                                                let pp_dump = json!({
                                                    "goals": [{
                                                        "pretty": format!("{}\n⊢ {}", hyps_texts.join("\n"), target),
                                                        "hyps": hyps_texts.iter().take(48).map(|s| json!({"text": s})).collect::<Vec<_>>()
                                                    }]
                                                });
                                                if let Some(script) =
                                                    plc::smt_lia::smt2_script_from_pp_dump(
                                                        &pp_dump,
                                                        smt_timeout_ms,
                                                        smt_seed,
                                                        smt_depth,
                                                    )
                                                {
                                                    let root = smt_dump_root(
                                                        &repo_root,
                                                        &smt_dump_dir_opt,
                                                    );
                                                    let path = maybe_write_smt2_dump(
                                                        &root, sk, goal_sig, smt_depth, &script,
                                                    );
                                                    smt_dumps_written += 1;
                                                    smt_dump_paths.push(path.clone());
                                                    smt_dump_last_path = Some(path.clone());
                                                    smt_dump_last_chars =
                                                        Some(script.chars().count());
                                                    smt_dump_last_preview =
                                                        Some(truncate_str(&script, 4_000));
                                                }
                                            }
                                            (
                                                None,
                                                Some(json!({
                                                    "entails": serde_json::Value::Null,
                                                    "source": "computed_unknown",
                                                    "state_key": sk,
                                                    "goal_sig": goal_sig,
                                                    "reason_unknown": smt_reuse
                                                        .as_ref()
                                                        .map(|s| s.stats())
                                                        .and_then(|v| v.get("last_reason_unknown").cloned())
                                                        .unwrap_or(serde_json::Value::Null),
                                                })),
                                            )
                                        }
                                    }
                                }
                            } else {
                                // No disk cache: still try a purely in-memory SMT check (no extra Lean calls).
                                let goal_sig = hash_text(&target);
                                let ck = (sk, goal_sig, smt_depth);
                                smt_cache_misses += 1;
                                let hyps_texts =
                                    goal_dump_hyps_cache.get(&k).cloned().unwrap_or_default();
                                if hyps_texts.is_empty() {
                                    (
                                        None,
                                        Some(json!({
                                            "entails": serde_json::Value::Null,
                                            "source": "no_hyps",
                                            "state_key": sk,
                                            "goal_sig": goal_sig,
                                        })),
                                    )
                                } else {
                                    let t0 = std::time::Instant::now();
                                    let (ent, attempts) = smt_entails_from_hyps_target_escalating(
                                        &hyps_texts,
                                        &target,
                                        smt_timeout_ms,
                                        smt_seed,
                                        smt_depth,
                                        &smt_solver_norm,
                                        smt_aggressive,
                                        &mut smt_reuse,
                                        &mut smt_entails_trace,
                                    )
                                    .unwrap_or_else(|e| {
                                        smt_errors += 1;
                                        smt_last_error = Some(truncate_str(&e, 400));
                                        (None, 1)
                                    });
                                    smt_entails_attempts =
                                        smt_entails_attempts.saturating_add(attempts);
                                    if attempts > 1 {
                                        smt_entails_escalations = smt_entails_escalations
                                            .saturating_add(attempts.saturating_sub(1));
                                    }
                                    prof_smt_ms =
                                        prof_smt_ms.saturating_add(t0.elapsed().as_millis() as u64);
                                    if let Some(ent) = ent {
                                        smt_entails_cache.insert(ck, ent);
                                        let mut hint = json!({
                                            "entails": ent,
                                            "source": "computed",
                                            "state_key": sk,
                                            "goal_sig": goal_sig,
                                        });
                                        if smt_dump && smt_dumps_written < smt_dump_max as u64 {
                                            let pp_dump = json!({
                                                "goals": [{
                                                    "pretty": format!("{}\n⊢ {}", hyps_texts.join("\n"), target),
                                                    "hyps": hyps_texts.iter().take(48).map(|s| json!({"text": s})).collect::<Vec<_>>()
                                                }]
                                            });
                                            if let Some(script) =
                                                plc::smt_lia::smt2_script_from_pp_dump(
                                                    &pp_dump,
                                                    smt_timeout_ms,
                                                    smt_seed,
                                                    smt_depth,
                                                )
                                            {
                                                let root =
                                                    smt_dump_root(&repo_root, &smt_dump_dir_opt);
                                                let path = maybe_write_smt2_dump(
                                                    &root, sk, goal_sig, smt_depth, &script,
                                                );
                                                smt_dumps_written += 1;
                                                smt_dump_paths.push(path.clone());
                                                smt_dump_last_path = Some(path.clone());
                                                smt_dump_last_chars = Some(script.chars().count());
                                                smt_dump_last_preview =
                                                    Some(truncate_str(&script, 4_000));
                                                if let Some(obj) = hint.as_object_mut() {
                                                    obj.insert(
                                                        "smt2_dump".to_string(),
                                                        json!(path),
                                                    );
                                                    obj.insert(
                                                        "smt2_dump_chars".to_string(),
                                                        json!(script.chars().count()),
                                                    );
                                                    obj.insert(
                                                        "smt2_dump_preview".to_string(),
                                                        json!(truncate_str(&script, 2_000)),
                                                    );
                                                }
                                            }
                                        }
                                        (Some(ent), Some(hint))
                                    } else {
                                        if smt_dump && smt_dumps_written < smt_dump_max as u64 {
                                            let pp_dump = json!({
                                                "goals": [{
                                                    "pretty": format!("{}\n⊢ {}", hyps_texts.join("\n"), target),
                                                    "hyps": hyps_texts.iter().take(48).map(|s| json!({"text": s})).collect::<Vec<_>>()
                                                }]
                                            });
                                            if let Some(script) =
                                                plc::smt_lia::smt2_script_from_pp_dump(
                                                    &pp_dump,
                                                    smt_timeout_ms,
                                                    smt_seed,
                                                    smt_depth,
                                                )
                                            {
                                                let root =
                                                    smt_dump_root(&repo_root, &smt_dump_dir_opt);
                                                let path = maybe_write_smt2_dump(
                                                    &root, sk, goal_sig, smt_depth, &script,
                                                );
                                                smt_dumps_written += 1;
                                                smt_dump_paths.push(path.clone());
                                                smt_dump_last_path = Some(path.clone());
                                                smt_dump_last_chars = Some(script.chars().count());
                                                smt_dump_last_preview =
                                                    Some(truncate_str(&script, 4_000));
                                            }
                                        }
                                        (
                                            None,
                                            Some(json!({
                                                "entails": serde_json::Value::Null,
                                                "source": "computed_unknown",
                                                "state_key": sk,
                                                "goal_sig": goal_sig,
                                                "reason_unknown": smt_reuse
                                                    .as_ref()
                                                    .map(|s| s.stats())
                                                    .and_then(|v| v.get("last_reason_unknown").cloned())
                                                    .unwrap_or(serde_json::Value::Null),
                                            })),
                                        )
                                    }
                                }
                            }
                        } else {
                            (None, None)
                        }
                    } else {
                        (None, None)
                    };

                    // Conservative default: when using depth-limited SMT, a SAT result can simply mean
                    // "not enough relevant hyps were included". Treat `false` as `unknown` unless we
                    // used the full hypothesis set (`smt_depth==0`).
                    let smt_entails_effective: Option<bool> =
                        if smt_depth > 0 && smt_entails_opt == Some(false) {
                            None
                        } else {
                            smt_entails_opt
                        };

                    // If requested, attach a small, bounded “fragment explanation” (what SMT actually
                    // saw after depth filtering). This is designed to be cheap (no extra Lean calls).
                    if smt_explain {
                        if let Some(v) = smt_hint_json.as_mut().and_then(|v| v.as_object_mut()) {
                            v.insert("depth".to_string(), json!(smt_depth));
                            // Best-effort: compute from cached hyps_texts + target.
                            let th = hash_text(&parent.text);
                            let k = (th, parent.text.len(), sel.line);
                            let target = goal_dump_cache
                                .get(&k)
                                .map(|(_, _, _, t)| t.clone())
                                .unwrap_or_default();
                            let hyps_texts =
                                goal_dump_hyps_cache.get(&k).cloned().unwrap_or_default();
                            v.insert(
                                "fragment".to_string(),
                                smt_explain_fragment_from_hyps_target(
                                    &hyps_texts,
                                    &target,
                                    smt_depth,
                                    smt_explain_max_hyps,
                                )
                                .unwrap_or(serde_json::Value::Null),
                            );

                            // Optional unsat core / support set (expensive): only meaningful when entailment is `true`.
                            if (smt_unsat_core || smt_support) && smt_entails_opt == Some(true) {
                                let mut pretty = String::new();
                                for h in hyps_texts.iter().take(48) {
                                    pretty.push_str(h);
                                    pretty.push('\n');
                                }
                                pretty.push_str("⊢ ");
                                pretty.push_str(&target);
                                let hyps_json: Vec<serde_json::Value> = hyps_texts
                                    .iter()
                                    .take(48)
                                    .map(|s| json!({ "text": s }))
                                    .collect();
                                let pp_dump = json!({
                                    "goals": [{
                                        "pretty": pretty,
                                        "hyps": hyps_json
                                    }]
                                });
                                let core = plc::smt_lia::unsat_core_from_pp_dump(
                                    &pp_dump,
                                    smt_timeout_ms,
                                    smt_seed,
                                    smt_depth,
                                    if smt_unsat_core {
                                        smt_unsat_core_max
                                    } else {
                                        smt_support_max
                                    },
                                )
                                .unwrap_or(None)
                                .unwrap_or(serde_json::Value::Null);
                                v.insert("unsat_core".to_string(), core);
                                let ctx_hyp_names: std::collections::HashSet<String> = hyps_texts
                                    .iter()
                                    .filter_map(|s| hyp_name_from_text_line(s))
                                    .collect();
                                if let Some(hs) = smt_support_lean_hints(
                                    v.get("unsat_core").unwrap(),
                                    &ctx_hyp_names,
                                ) {
                                    v.insert("smt_support_lean".to_string(), json!(hs));
                                }
                            }
                        }
                    }

                    // If SMT says the local LIA fragment entails the target, inject a few
                    // deterministic arithmetic tactics at the front. This makes the SMT signal
                    // *actionable* rather than only a ranking hint.
                    if smt_entails_effective == Some(true) {
                        // If we computed an explicit support-guided Lean proof sketch, try it first.
                        // (It will be adapted to tactic-vs-term context below.)
                        let mut support_candidates: Vec<String> = smt_hint_json
                            .as_ref()
                            .and_then(|v| v.get("smt_support_lean"))
                            .and_then(|v| v.as_array())
                            .map(|xs| {
                                xs.iter()
                                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();
                        if !support_candidates.is_empty() {
                            // Adapt all candidates to term-vs-tactic context (bounded).
                            support_candidates.truncate(smt_support_max);
                            support_candidates = adapt_candidates_for_sorry_context(
                                &support_candidates,
                                &sel.line_text,
                                is_tactic_context,
                            );
                        }
                        // Candidates are proof-term replacements inside an existing `by` block,
                        // so these are tactics, not `by ...`.
                        let mut inject: Vec<String> = Vec::new();
                        inject.extend(support_candidates.into_iter().take(smt_support_max));
                        inject.extend(vec![
                            "omega".to_string(),
                            "linarith".to_string(),
                            "nlinarith".to_string(),
                            "norm_cast".to_string(),
                            "simp; omega".to_string(),
                        ]);
                        // Prepend and then sanitize/dedupe/bound later.
                        inject.extend(candidates_here.into_iter());
                        candidates_here = sanitize_candidates(inject);
                    } else if smt_entails_effective == Some(false) && smt_depth == 0 {
                        // If SMT says the LIA fragment does *not* entail the target, avoid spending
                        // node budget on pure arithmetic tactics that are unlikely to close the goal.
                        // (This is intentionally conservative: only drop obvious LIA-only candidates.)
                        candidates_here.retain(|c| {
                            let t = c.to_lowercase();
                            !(t.contains("omega")
                                || t.contains("linarith")
                                || t.contains("nlinarith")
                                || t.contains("norm_num"))
                        });
                        candidates_here = sanitize_candidates(candidates_here);
                    }

                    // If the goal looks “wide” or “context heavy”, we should prefer low-branching
                    // candidates even more aggressively.
                    let (n_goals, hyps_total, meta_vars_target) = {
                        let th = hash_text(&parent.text);
                        let k = (th, parent.text.len(), sel.line);
                        goal_dump_cache
                            .get(&k)
                            .map(|(_, ng, ht, target)| {
                                let meta =
                                    target.matches("?m").count() + target.matches("?_").count();
                                (*ng as i64, *ht as i64, meta as i64)
                            })
                            .unwrap_or((0, 0, 0))
                    };

                    // Rank candidates and keep a compact explanation for grokkability.
                    #[derive(Clone)]
                    struct CandRank {
                        cand: String,
                        cand_h: u64,
                        prior: i64,
                        complexity: i64,
                        smt_bonus: i64,
                        shape_bonus: i64,
                        is_arith: bool,
                        arith_keyword: Option<String>,
                        first_cmd: String,
                        category: &'static str,
                    }
                    let first_cmd_of_candidate = |cand: &str| -> String {
                        // Best-effort: extract the first “interesting” tactic token.
                        //
                        // This is intentionally shallow: it's used only for *ranking hints* and to
                        // decide whether SMT entailment should bias toward/away from arithmetic tactics.
                        // It is *not* used to decide correctness, and it must never trigger extra Lean calls.
                        //
                        // Examples:
                        // - "by\n  (simp; done)" -> "simp"
                        // - "by\n  classical\n  first | (simp; done) | ... " -> "first"
                        // - "(omega; done)" -> "omega"
                        for ln in cand.lines() {
                            let mut t = ln.trim();
                            if t.is_empty() {
                                continue;
                            }
                            // common scaffolding lines in our candidates
                            if t == "by" {
                                continue;
                            }
                            if t == "classical" || t == "all_goals" {
                                continue;
                            }
                            if t == "·" {
                                continue;
                            }
                            // Strip one layer of parentheses (common in our candidates).
                            if t.starts_with('(') && t.ends_with(')') && t.len() >= 2 {
                                t = &t[1..t.len() - 1];
                                t = t.trim();
                            }
                            // Tokenize: read alnum/_/?/- characters.
                            let mut out = String::new();
                            for ch in t.chars() {
                                if ch.is_alphanumeric() || ch == '_' || ch == '?' || ch == '-' {
                                    out.push(ch);
                                } else {
                                    break;
                                }
                            }
                            if !out.is_empty() {
                                return out;
                            }
                        }
                        "".to_string()
                    };
                    let arith_keyword_of = |cand: &str, first_cmd: &str| -> Option<String> {
                        // Prefer “first command” classification, then fall back to substring match.
                        // Keep this conservative: it’s only a ranking hint, not proof logic.
                        //
                        // Why only these:
                        // - these tactics are "high leverage" when the goal is linear arithmetic,
                        // - and "high waste" when it isn't (they consume attempts without progress).
                        // Everything else is treated as non-arithmetic for the purpose of the SMT bias.
                        const ARITH: [&str; 4] = ["omega", "linarith", "nlinarith", "zify"];
                        for k in ARITH {
                            if first_cmd == k || first_cmd == format!("{k}?") {
                                return Some(k.to_string());
                            }
                        }
                        for k in ARITH {
                            if cand.contains(k) {
                                return Some(k.to_string());
                            }
                        }
                        None
                    };
                    let goal_shape_even_div: bool = goal_dump_v
                        .as_ref()
                        .and_then(|gd| {
                            gd.get("pp_dump")
                                .and_then(|v| v.get("goals"))
                                .and_then(|v| v.as_array())
                                .and_then(|a| a.first())
                        })
                        .map(|g0| {
                            let target = g0
                                .get("pretty")
                                .and_then(|v| v.as_str())
                                .and_then(|s| {
                                    s.lines().find_map(|ln| {
                                        ln.trim_start()
                                            .strip_prefix("⊢")
                                            .map(|r| r.trim().to_string())
                                    })
                                })
                                .unwrap_or_default();
                            let has_even = g0
                                .get("hyps")
                                .and_then(|v| v.as_array())
                                .map(|hs| {
                                    hs.iter().any(|h| {
                                        h.get("text")
                                            .and_then(|v| v.as_str())
                                            .map(|s| s.contains("Even"))
                                            .unwrap_or(false)
                                    })
                                })
                                .unwrap_or(false);
                            target.contains("∣") && has_even
                        })
                        .unwrap_or(false);

                    let mut ranked: Vec<CandRank> = candidates_here
                        .iter()
                        .take(cap)
                        .map(|c0| {
                            let c = c0.clone();
                            let cand_h = hash_text(&c);
                            let prior = state_key_opt
                                .and_then(|sk| state_action_cache.get(&(sk, cand_h)).copied())
                                .unwrap_or(50_000) as i64;
                            let first_cmd = first_cmd_of_candidate(&c);
                            let arith_keyword = arith_keyword_of(&c, first_cmd.as_str());
                            let is_arith = arith_keyword.is_some();
                            let category: &'static str = if is_arith {
                                "lia"
                            } else if first_cmd.starts_with("simp") {
                                "simpish"
                            } else if first_cmd.starts_with("aesop")
                                || first_cmd == "first"
                                || first_cmd == "exact?"
                                || first_cmd == "apply?"
                            {
                                "searchish"
                            } else if first_cmd.starts_with("ring") || first_cmd == "norm_num" {
                                "ringish"
                            } else if first_cmd.is_empty() {
                                "unknown"
                            } else {
                                "other"
                            };
                            let smt_bonus: i64 = match smt_entails_effective {
                                Some(true) if is_arith => -5_000,
                                Some(false) if is_arith => 5_000,
                                _ => 0,
                            };
                            // Note: `smt_bonus` is deliberately coarse. We only want SMT to act as a
                            // *tie-breaker* or a strong nudge, not to dominate the learned/state-action prior.
                            let shape_bonus: i64 = if goal_shape_even_div && first_cmd == "rcases" {
                                -50_000
                            } else {
                                0
                            };
                            let lines = c.lines().count() as i64;
                            let holes = c.matches("?_").count() as i64;
                            let bullets = c.matches("\n·").count() as i64;
                            let len = c.chars().count() as i64;
                            let complexity = holes * (50 + 10 * n_goals)
                                + bullets * 10
                                + lines * 5
                                + len / 40
                                + hyps_total / 20
                                + shape_bonus;
                            CandRank {
                                cand: c,
                                cand_h,
                                prior,
                                complexity,
                                smt_bonus,
                                shape_bonus,
                                is_arith,
                                arith_keyword,
                                first_cmd,
                                category,
                            }
                        })
                        .collect();
                    prof_candidates_considered += ranked.len() as u64;
                    ranked.sort_by_key(|r| (r.prior, r.complexity + r.smt_bonus));
                    // Avoid retrying known-bad actions on the same goal state within this run.
                    // This is safe because it only filters actions we already executed (and scored) for
                    // the same `state_key` + candidate hash.
                    let state_action_skip = std::env::var("PROOFPATCH_STATE_ACTION_SKIP")
                        .ok()
                        .map(|s| {
                            matches!(
                                s.trim().to_lowercase().as_str(),
                                "0" | "false" | "no" | "n" | "off"
                            )
                        })
                        .map(|disabled| !disabled)
                        .unwrap_or(true);
                    let mut cand_vec: Vec<String> = Vec::new();
                    let mut skipped_by_state_action: u64 = 0;
                    for r in ranked.iter() {
                        if state_action_skip && state_key_opt.is_some() && r.prior >= 200_000 {
                            // Failed actions are intentionally stored worse than unknown; skipping them saves work.
                            skipped_by_state_action += 1;
                            continue;
                        }
                        cand_vec.push(r.cand.clone());
                    }
                    if cand_vec.is_empty() {
                        // Never allow the filter to eliminate everything.
                        cand_vec = ranked.iter().map(|r| r.cand.clone()).collect();
                    }
                    record_event(
                        "candidates_filtered",
                        json!({
                            "line": sel.line,
                            "state_key": state_key_opt,
                            "skip_enabled": state_action_skip,
                            "ranked_n": ranked.len(),
                            "kept_n": cand_vec.len(),
                            "skipped_by_state_action": skipped_by_state_action,
                        }),
                    );

                    let rank_hint_json = {
                        let top_k = 6usize;
                        let mut top_rows: Vec<serde_json::Value> = Vec::new();
                        for r in ranked.iter().take(top_k) {
                            let preview = r
                                .cand
                                .lines()
                                .next()
                                .unwrap_or("")
                                .trim()
                                .chars()
                                .take(120)
                                .collect::<String>();
                            let smt_reason = if meta_vars_target > 0 && r.is_arith {
                                "meta_vars_target"
                            } else if r.smt_bonus != 0 && r.is_arith {
                                match smt_entails_effective {
                                    Some(true) => "smt_entails_true_boost",
                                    Some(false) => "smt_entails_false_penalize",
                                    None => "no_smt_signal",
                                }
                            } else if r.is_arith {
                                match smt_entails_effective {
                                    Some(true) | Some(false) => "smt_signal_no_effect",
                                    None => "no_smt_signal",
                                }
                            } else {
                                "non_lia"
                            };
                            top_rows.push(json!({
                                "cand_hash": r.cand_h,
                                "preview": preview,
                                "is_arith": r.is_arith,
                                "arith_keyword": r.arith_keyword,
                                "first_cmd": r.first_cmd,
                                "category": r.category,
                                "prior": r.prior,
                                "complexity": r.complexity,
                                "smt_bonus": r.smt_bonus,
                                "shape_bonus": r.shape_bonus,
                                "second_key": r.complexity + r.smt_bonus,
                                "reason": smt_reason,
                            }));
                        }
                        Some(json!({
                            "line": sel.line,
                            "n_goals": n_goals,
                            "hyps_total": hyps_total,
                            "meta_vars_target": meta_vars_target,
                            // Canonical, stable field for SMT provenance/artifacts.
                            "smt_evidence": smt_hint_json.clone().unwrap_or(serde_json::Value::Null),
                            // Back-compat alias (older outputs used `rank_hint.smt`).
                            "smt": smt_hint_json.clone().unwrap_or(serde_json::Value::Null),
                            "top": top_rows,
                        }))
                    };

                    for cand in cand_vec.iter().take(verify_k) {
                        prof_candidates_verified += 1;
                        if all.len() + new_frontier.len() >= max_nodes {
                            break;
                        }
                        let t0 = std::time::Instant::now();
                        let patched = plc::patch_first_sorry_in_region(
                            &parent.text,
                            region.0,
                            region.1,
                            cand,
                        );
                        let Ok(patched) = patched else {
                            continue;
                        };
                        prof_patch_ms =
                            prof_patch_ms.saturating_add(t0.elapsed().as_millis() as u64);
                        // Optional bounded rollout: patch the next few closest sorries using a safe fill script.
                        // This makes each expansion behave more like a mini proof-tree step rather than
                        // “one hole per depth”.
                        let mut rolled_text = patched.text.clone();
                        let mut rolled_line = patched.line;
                        let mut roll_last_region: Option<(usize, usize)> = Some(region);
                        let mut roll_last_replacement: Option<String> = Some(cand.clone());
                        for _ in 0..rollout_k {
                            let t0 = std::time::Instant::now();
                            let locs_r = plc::locate_sorries_in_text(&rolled_text, 200, 1)
                                .unwrap_or_default();
                            prof_locate_sorries_ms = prof_locate_sorries_ms
                                .saturating_add(t0.elapsed().as_millis() as u64);
                            if locs_r.is_empty() {
                                break;
                            }
                            let next = locs_r
                                .iter()
                                .min_by_key(|l| (l.line as i64 - rolled_line as i64).abs())
                                .cloned();
                            let Some(sel_r) = next else {
                                break;
                            };

                            // Detect tactic context for this `sorry` (including bullet/case).
                            let is_tactic_context_r = {
                                let lines: Vec<&str> = rolled_text.lines().collect();
                                let line_idx0 = sel_r.line.saturating_sub(1);
                                let indent = sel_r
                                    .line_text
                                    .chars()
                                    .take_while(|c| *c == ' ' || *c == '\t')
                                    .count();
                                let mut found = false;
                                if line_idx0 < lines.len() {
                                    let start = line_idx0.saturating_sub(80);
                                    for k in (start..line_idx0).rev() {
                                        let l = lines[k];
                                        let t = l.trim();
                                        if t.is_empty() || t.starts_with("--") {
                                            continue;
                                        }
                                        if t == "by"
                                            || t.contains(":= by")
                                            || t.contains("=> by")
                                            || t.starts_with('·')
                                            || t.starts_with("case ")
                                        {
                                            found = true;
                                            break;
                                        }
                                        let ind_k = l
                                            .chars()
                                            .take_while(|c| *c == ' ' || *c == '\t')
                                            .count();
                                        if ind_k < indent
                                            && (t.starts_with("lemma ")
                                                || t.starts_with("theorem ")
                                                || t.starts_with("def ")
                                                || t.starts_with("instance ")
                                                || t.starts_with("structure "))
                                        {
                                            break;
                                        }
                                    }
                                }
                                found
                            };

                            let fill_repl = if is_tactic_context_r {
                                // Tactic hole: single-line `first | ... | sorry` is robust in nested contexts.
                                safe_first_line()
                            } else {
                                // Term hole: a proof term.
                                "by\n  first | (simp; done) | (aesop; done) | (omega; done) | (nlinarith; done) | (linarith; done) | (ring_nf; done) | (norm_num; done) | sorry".to_string()
                            };

                            let patched_r = plc::patch_first_sorry_in_region(
                                &rolled_text,
                                sel_r.region_start,
                                sel_r.region_end,
                                &fill_repl,
                            );
                            let Ok(patched_r) = patched_r else {
                                break;
                            };
                            rolled_text = patched_r.text;
                            rolled_line = patched_r.line;
                            roll_last_region = Some((sel_r.region_start, sel_r.region_end));
                            roll_last_replacement = Some(fill_repl);
                        }

                        // Evaluate.
                        let h = hash_text(&rolled_text);
                        let mut verify_ms: u64 = 0;
                        let mut verify_cache: &'static str = "mem";
                        let (raw_v, summary, locs2_len, conservative2) = if let Some(c) =
                            eval_cache.get(&h).filter(|c| c.len == rolled_text.len())
                        {
                            (
                                c.verify_raw.clone(),
                                c.verify_summary.clone(),
                                c.sorries,
                                c.conservative_sorries,
                            )
                        } else {
                            if let Some(cd) = cache_dir.as_ref() {
                                if let Some((raw_v, summary, sorries, conservative)) =
                                    cache_read_eval(cd, h, rolled_text.len())
                                {
                                    verify_cache = "disk";
                                    disk_cache_eval_hits += 1;
                                    eval_cache.insert(
                                        h,
                                        CachedEval {
                                            len: rolled_text.len(),
                                            verify_raw: raw_v.clone(),
                                            verify_summary: summary.clone(),
                                            sorries,
                                            conservative_sorries: conservative,
                                        },
                                    );
                                    (raw_v, summary, sorries, conservative)
                                } else {
                                    disk_cache_eval_misses += 1;
                                    verify_cache = "none";
                                    let Some(dur) = budget_dur(timeout_s) else {
                                        bailed_total_timeout = true;
                                        record_event(
                                            "bailout_total_timeout",
                                            json!({ "where": "verify_node" }),
                                        );
                                        break;
                                    };
                                    let t0 = std::time::Instant::now();
                                    let raw = rt
                                        .block_on(plc::verify_lean_text(
                                            &repo_root,
                                            &rolled_text,
                                            dur,
                                        ))
                                        .unwrap_or(plc::VerifyResult {
                                            ok: false,
                                            timeout: false,
                                            returncode: None,
                                            stdout: String::new(),
                                            stderr: "verify failed".to_string(),
                                            cmd: vec![],
                                            cwd: repo_root.display().to_string(),
                                            tmp_file: None,
                                        });
                                    verify_ms = t0.elapsed().as_millis() as u64;
                                    prof_verify_nodes_ms =
                                        prof_verify_nodes_ms.saturating_add(verify_ms);
                                    prof_verify_nodes_calls += 1;
                                    let raw_v = serde_json::to_value(raw)
                                        .map_err(|e| format!("serialize verify: {e}"))?;
                                    let summary = verify_summary_from_raw_value(&raw_v);
                                    let t0 = std::time::Instant::now();
                                    let locs2 = plc::locate_sorries_in_text(&rolled_text, 500, 1)
                                        .unwrap_or_default();
                                    prof_locate_sorries_ms = prof_locate_sorries_ms
                                        .saturating_add(t0.elapsed().as_millis() as u64);
                                    let t0 = std::time::Instant::now();
                                    let conservative2 =
                                        plc::count_sorry_tokens_conservative(&rolled_text)
                                            .unwrap_or(0);
                                    prof_conservative_sorries_ms = prof_conservative_sorries_ms
                                        .saturating_add(t0.elapsed().as_millis() as u64);
                                    eval_cache.insert(
                                        h,
                                        CachedEval {
                                            len: rolled_text.len(),
                                            verify_raw: raw_v.clone(),
                                            verify_summary: summary.clone(),
                                            sorries: locs2.len(),
                                            conservative_sorries: conservative2,
                                        },
                                    );
                                    cache_write_eval(
                                        cd,
                                        h,
                                        rolled_text.len(),
                                        &raw_v,
                                        &summary,
                                        locs2.len(),
                                        conservative2,
                                    );
                                    (raw_v, summary, locs2.len(), conservative2)
                                }
                            } else {
                                verify_cache = "none";
                                let Some(dur) = budget_dur(timeout_s) else {
                                    bailed_total_timeout = true;
                                    record_event(
                                        "bailout_total_timeout",
                                        json!({ "where": "verify_node" }),
                                    );
                                    break;
                                };
                                let t0 = std::time::Instant::now();
                                let raw = rt
                                    .block_on(plc::verify_lean_text(&repo_root, &rolled_text, dur))
                                    .unwrap_or(plc::VerifyResult {
                                        ok: false,
                                        timeout: false,
                                        returncode: None,
                                        stdout: String::new(),
                                        stderr: "verify failed".to_string(),
                                        cmd: vec![],
                                        cwd: repo_root.display().to_string(),
                                        tmp_file: None,
                                    });
                                verify_ms = t0.elapsed().as_millis() as u64;
                                prof_verify_nodes_ms =
                                    prof_verify_nodes_ms.saturating_add(verify_ms);
                                prof_verify_nodes_calls += 1;
                                let raw_v = serde_json::to_value(raw)
                                    .map_err(|e| format!("serialize verify: {e}"))?;
                                let summary = verify_summary_from_raw_value(&raw_v);
                                let t0 = std::time::Instant::now();
                                let locs2 = plc::locate_sorries_in_text(&rolled_text, 500, 1)
                                    .unwrap_or_default();
                                prof_locate_sorries_ms = prof_locate_sorries_ms
                                    .saturating_add(t0.elapsed().as_millis() as u64);
                                let t0 = std::time::Instant::now();
                                let conservative2 =
                                    plc::count_sorry_tokens_conservative(&rolled_text).unwrap_or(0);
                                prof_conservative_sorries_ms = prof_conservative_sorries_ms
                                    .saturating_add(t0.elapsed().as_millis() as u64);
                                eval_cache.insert(
                                    h,
                                    CachedEval {
                                        len: rolled_text.len(),
                                        verify_raw: raw_v.clone(),
                                        verify_summary: summary.clone(),
                                        sorries: locs2.len(),
                                        conservative_sorries: conservative2,
                                    },
                                );
                                (raw_v, summary, locs2.len(), conservative2)
                            }
                        };
                        record_event(
                            "verify_node",
                            json!({
                                "node_id": next_id,
                                "parent_id": parent.id,
                                "depth": parent.depth + 1,
                                "cache": verify_cache,
                                "ms": if verify_cache == "none" { json!(verify_ms) } else { serde_json::Value::Null },
                                "ok": summary.get("ok").and_then(|v| v.as_bool()).unwrap_or(false),
                                "counts": summary.get("counts").cloned().unwrap_or(serde_json::Value::Null),
                                "sorries": locs2_len,
                                "conservative_sorries": conservative2,
                                "replacement_hash": if log_level >= 2 { json!(hash_text(cand)) } else { serde_json::Value::Null },
                                "replacement_preview": if log_level >= 2 {
                                    let t = cand.trim();
                                    let head: String = t.chars().take(240).collect();
                                    if t.chars().count() > 240 { json!(format!("{head}...")) } else { json!(head) }
                                } else {
                                    serde_json::Value::Null
                                },
                            }),
                        );

                        // Record state-action outcome for this hole if we can key it by goal-state.
                        if let Some(sk) = state_key_opt {
                            let cand_h = hash_text(cand);
                            let ok = summary.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
                            let errs = summary
                                .get("counts")
                                .and_then(|c| c.get("errors"))
                                .and_then(|v| v.as_u64())
                                .unwrap_or(999) as i32;
                            let sw = summary
                                .get("counts")
                                .and_then(|c| c.get("sorry_warnings"))
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0) as i32;
                            let score = if ok {
                                errs + sw * 10 + (locs2_len as i32)
                            } else {
                                // Important: scores are used as a *prior* where lower is better.
                                // Failed actions should therefore be *worse* than the default unknown prior (50_000).
                                200_000 + errs + sw * 10 + (locs2_len as i32)
                            };
                            state_action_cache
                                .entry((sk, cand_h))
                                .and_modify(|v| *v = (*v).min(score))
                                .or_insert(score);
                        }

                        // Best-effort goal signature for this hole (used to stabilize which branch we keep working on).
                        let focus_goal_sig = {
                            let th = hash_text(&parent.text);
                            let k = (th, parent.text.len(), sel.line);
                            goal_dump_cache
                                .get(&k)
                                .and_then(|(_, _, _, tgt)| {
                                    if tgt.is_empty() {
                                        None
                                    } else {
                                        Some(hash_text(tgt))
                                    }
                                })
                                .or(parent.focus_goal_sig)
                        };
                        new_frontier.push(Node {
                            id: next_id,
                            depth: parent.depth + 1,
                            text: rolled_text.clone(),
                            focus_decl_name: parent
                                .focus_decl_name
                                .clone()
                                .or_else(|| sel.decl_name.clone()),
                            focus_line: Some(rolled_line),
                            focus_goal_sig,
                            last_region: roll_last_region,
                            last_replacement: roll_last_replacement,
                            parent_id: Some(parent.id),
                            verify_raw: Some(raw_v),
                            verify_summary: Some(summary),
                            sorries: Some(locs2_len),
                            conservative_sorries: Some(conservative2),
                            smt_hint: smt_hint_json.clone(),
                            rank_hint: rank_hint_json.clone(),
                        });
                        next_id += 1;
                    }
                }
                if bailed_total_timeout {
                    break;
                }
                frontier = new_frontier;
            }

            // Pick best node (either done or best-scoring leaf).
            let best = if let Some(ref d) = best_done {
                d.clone()
            } else if all.is_empty() {
                // If we bailed early (e.g. total-timeout) before moving any nodes into `all`,
                // fall back to the root node so we can still emit a useful partial result.
                frontier.first().cloned().unwrap_or_else(|| Node {
                    id: 0,
                    depth: 0,
                    text: original_text.clone(),
                    focus_decl_name: focus_decl_name.clone(),
                    focus_line: focus_line_1,
                    focus_goal_sig: None,
                    last_region: None,
                    last_replacement: None,
                    parent_id: None,
                    verify_raw: Some(baseline_raw_v.clone()),
                    verify_summary: Some(baseline_summary.clone()),
                    sorries: Some(
                        plc::locate_sorries_in_text(&original_text, 500, 1)
                            .unwrap_or_default()
                            .len(),
                    ),
                    conservative_sorries: Some(
                        plc::count_sorry_tokens_conservative(&original_text).unwrap_or(0),
                    ),
                    smt_hint: None,
                    rank_hint: None,
                })
            } else {
                let mut xs = all.clone();
                xs.sort_by(|a, b| {
                    let sa = a.verify_summary.as_ref().unwrap();
                    let sb = b.verify_summary.as_ref().unwrap();
                    let ka = verify_score_key(
                        sa,
                        a.sorries.unwrap_or(999),
                        a.conservative_sorries.unwrap_or(999),
                    );
                    let kb = verify_score_key(
                        sb,
                        b.sorries.unwrap_or(999),
                        b.conservative_sorries.unwrap_or(999),
                    );
                    ka.cmp(&kb).then_with(|| a.id.cmp(&b.id))
                });
                xs[0].clone()
            };

            // Also compute a "best progress" node that prefers fewer remaining sorries even
            // when compilation is failing (useful for multi-step repair loops).
            let best_progress = if all.is_empty() {
                best.clone()
            } else {
                let mut xs = all.clone();
                xs.sort_by(|a, b| {
                    let sa = a.verify_summary.as_ref().unwrap();
                    let sb = b.verify_summary.as_ref().unwrap();
                    let ka = progress_score_key(
                        sa,
                        a.sorries.unwrap_or(999),
                        a.conservative_sorries.unwrap_or(999),
                    );
                    let kb = progress_score_key(
                        sb,
                        b.sorries.unwrap_or(999),
                        b.conservative_sorries.unwrap_or(999),
                    );
                    ka.cmp(&kb).then_with(|| a.id.cmp(&b.id))
                });
                xs[0].clone()
            };

            // `best` can be a solved node (`best_done`) that is not necessarily present in `all`.
            // When selecting `best-ok`, always consider `best` itself.
            let mut best_ok_pool: Vec<Node> = all
                .iter()
                .filter(|n| {
                    n.verify_summary
                        .as_ref()
                        .and_then(|s| s.get("ok"))
                        .and_then(|v| v.as_bool())
                        == Some(true)
                })
                .cloned()
                .collect();
            best_ok_pool.push(best.clone());
            // Dedup by id (prefer earlier entry).
            let mut seen = std::collections::HashSet::new();
            best_ok_pool.retain(|n| seen.insert(n.id));
            let best_ok = best_ok_pool
                .into_iter()
                .min_by_key(|n| (n.sorries.unwrap_or(999), n.id))
                .unwrap_or_else(|| best.clone());

            let pick = pick.trim().to_lowercase();
            let picked = match pick.as_str() {
                "best" => &best,
                "best-ok" => &best_ok,
                "best-progress" => &best_progress,
                other => {
                    return Err(format!(
                        "unknown --pick mode: {other} (expected best|best-ok|best-progress)"
                    ))
                }
            };
            let mut picked = picked.clone();

            // If the winning patch uses mathlib's suggestion tactics (`exact?`, `apply?`, etc.),
            // try to "determinize" it into the concrete suggestion (the `Try this:` line).
            //
            // Rationale: `exact?` is convenient but can drift across Mathlib/Lean versions; the
            // suggested `exact foo ...` term is much more stable.
            if let Some(repl) = picked.last_replacement.clone() {
                let uses_suggest = repl.contains("exact?")
                    || repl.contains("apply?")
                    || repl.contains("simp?")
                    || repl.contains("aesop?");
                if uses_suggest {
                    if let Some(raw) = picked.verify_raw.as_ref().and_then(|v| v.as_object()) {
                        let stdout = raw.get("stdout").and_then(|v| v.as_str()).unwrap_or("");
                        let stderr = raw.get("stderr").and_then(|v| v.as_str()).unwrap_or("");
                        let merged = format!("{stdout}\n{stderr}");
                        let suggs = plc::extract_try_this_suggestions(&merged);
                        if let Some(s0) = suggs.first() {
                            let s0_line = s0.trim();
                            // Build a replacement in the same syntactic "shape" as the original.
                            let repl_trim = repl.trim_start();
                            let is_term_hole = repl_trim.starts_with("by");
                            let indent_under_by = |block: &str| -> String {
                                block
                                    .lines()
                                    .map(|ln| format!("  {ln}"))
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            };
                            // First: if the suggestion is a single line, try to "inline determinize"
                            // within the existing replacement (e.g. rewrite `by exact?` into
                            // `by exact foo`), instead of replacing the entire block.
                            let nested_opt = if !s0_line.contains('\n') {
                                if repl.contains("exact?") && s0_line.starts_with("exact ") {
                                    Some(repl.replacen("exact?", s0_line, 1))
                                } else if repl.contains("apply?")
                                    && (s0_line.starts_with("apply ")
                                        || s0_line.starts_with("refine ")
                                        || s0_line.starts_with("exact "))
                                {
                                    Some(repl.replacen("apply?", s0_line, 1))
                                } else if repl.contains("have ")
                                    && repl.contains(":= by exact?")
                                    && s0_line.starts_with("exact ")
                                {
                                    Some(repl.replacen("exact?", s0_line, 1))
                                } else {
                                    None
                                }
                            } else {
                                None
                            };

                            let new_repl_opt = nested_opt.or_else(|| {
                                if is_term_hole {
                                    if s0.trim_start().starts_with("by") {
                                        Some(s0.to_string())
                                    } else if s0.contains('\n') {
                                        if repl.contains("classical") {
                                            Some(format!(
                                                "by\n  classical\n{}",
                                                indent_under_by(s0)
                                            ))
                                        } else {
                                            Some(format!("by\n{}", indent_under_by(s0)))
                                        }
                                    } else if repl.contains("classical") {
                                        Some(format!("by\n  classical\n  {s0}"))
                                    } else {
                                        Some(format!("by\n  {s0}"))
                                    }
                                } else {
                                    // Tactic hole: keep it single-line (indentation may vary by context).
                                    if s0.trim_start().starts_with("by") {
                                        None
                                    } else if s0.contains('\n') {
                                        None
                                    } else {
                                        Some(s0.to_string())
                                    }
                                }
                            });

                            if let Some(new_repl) = new_repl_opt.filter(|nr| nr != &repl) {
                                // Prefer rewriting within the recorded region (more robust than a
                                // global string match).
                                let det_text_opt = picked
                                    .last_region
                                    .and_then(|(a, b)| {
                                        // Token-level determinization inside the region.
                                        //
                                        // If we have multiple `Try this:` suggestions, we can often
                                        // eliminate multiple `exact?`/`apply?` occurrences in one pass.
                                        let mut t = picked.text.clone();
                                        let mut changed = false;
                                        for s in suggs.iter().take(6) {
                                            let s = s.trim();
                                            if s.is_empty() || s.contains('\n') {
                                                continue;
                                            }
                                            if t.contains("exact?")
                                                && repl.contains("exact?")
                                                && s.starts_with("exact ")
                                            {
                                                if let Some(t2) =
                                                    plc::tree_search::replace_in_region_first(
                                                        &t, a, b, "exact?", s,
                                                    )
                                                {
                                                    t = t2;
                                                    changed = true;
                                                    continue;
                                                }
                                            }
                                            if t.contains("apply?")
                                                && repl.contains("apply?")
                                                && (s.starts_with("apply ")
                                                    || s.starts_with("refine ")
                                                    || s.starts_with("exact "))
                                            {
                                                if let Some(t2) =
                                                    plc::tree_search::replace_in_region_first(
                                                        &t, a, b, "apply?", s,
                                                    )
                                                {
                                                    t = t2;
                                                    changed = true;
                                                    continue;
                                                }
                                            }
                                        }
                                        if changed {
                                            return Some(t);
                                        }

                                        // Fallback: replace the whole block (requires exact match).
                                        plc::tree_search::replace_in_region_once(
                                            &picked.text,
                                            a,
                                            b,
                                            &repl,
                                            &new_repl,
                                        )
                                    })
                                    .or_else(|| {
                                        // Fallback: only if we can locate the original replacement uniquely.
                                        if picked.text.matches(&repl).count() == 1 {
                                            Some(picked.text.replacen(&repl, &new_repl, 1))
                                        } else {
                                            None
                                        }
                                    });
                                if let Some(det_text) = det_text_opt {
                                    if let Some(dur) = budget_dur(timeout_s) {
                                        let t0 = std::time::Instant::now();
                                        if let Ok(v) = rt.block_on(plc::verify_lean_text(
                                            &repo_root, &det_text, dur,
                                        )) {
                                            let det_raw_v = serde_json::to_value(v)
                                                .map_err(|e| format!("serialize verify: {e}"))?;
                                            let det_summary =
                                                verify_summary_from_raw_value(&det_raw_v);
                                            let det_ok =
                                                det_summary.get("ok").and_then(|v| v.as_bool())
                                                    == Some(true);
                                            let det_sw = det_summary
                                                .get("counts")
                                                .and_then(|c| c.get("sorry_warnings"))
                                                .and_then(|v| v.as_u64())
                                                .unwrap_or(0);
                                            if det_ok && det_sw == 0 {
                                                // Accept determinized patch.
                                                picked.text = det_text;
                                                picked.last_replacement = Some(new_repl);
                                                picked.verify_raw = Some(det_raw_v);
                                                picked.verify_summary = Some(det_summary);
                                                let locs2 = plc::locate_sorries_in_text(
                                                    &picked.text,
                                                    500,
                                                    1,
                                                )
                                                .unwrap_or_default();
                                                picked.sorries = Some(locs2.len());
                                                picked.conservative_sorries = Some(
                                                    plc::count_sorry_tokens_conservative(
                                                        &picked.text,
                                                    )
                                                    .unwrap_or(0),
                                                );
                                            }
                                        }
                                        prof_verify_nodes_ms = prof_verify_nodes_ms
                                            .saturating_add(t0.elapsed().as_millis() as u64);
                                        prof_verify_nodes_calls += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Finalize event stream before rendering any human summaries.
            // We also drop the recorder closure so we can immutably read `events_tail` safely.
            record_event(
                "end",
                json!({
                    "bailouts": {
                        "total_timeout": bailed_total_timeout,
                    },
                    "remaining_ms": remaining_ms(run_deadline),
                }),
            );
            drop(record_event);

            // Hard-focus “stuck” signal: we refused to drift to other decls and found no more holes
            // inside the focus decl on at least one expanded node.
            let hard_focus_stuck = focus_decl_hard
                && !bailed_total_timeout
                && best_done.is_none()
                && events_by_kind
                    .get("focus_decl_hard_no_sorries_in_decl")
                    .copied()
                    .unwrap_or(0)
                    > 0;

            if !quiet && summary_level > 0 && log_level > 0 {
                let counts = |s: &serde_json::Value, k: &str| -> u64 {
                    s.get("counts")
                        .and_then(|c| c.get(k))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0)
                };
                let ok = |s: &serde_json::Value| -> bool {
                    s.get("ok").and_then(|v| v.as_bool()).unwrap_or(false)
                };

                let b_ok = ok(&baseline_summary);
                let b_errs = counts(&baseline_summary, "errors");
                let b_warns = counts(&baseline_summary, "warnings");
                let b_sw = counts(&baseline_summary, "sorry_warnings");

                let best_summary = best.verify_summary.as_ref().unwrap();
                let best_ok2 = ok(best_summary);
                let best_errs = counts(best_summary, "errors");
                let best_warns = counts(best_summary, "warnings");
                let best_sw = counts(best_summary, "sorry_warnings");
                let best_sorries = best.sorries.unwrap_or(999);

                let ps = picked.verify_summary.as_ref().unwrap();
                let pok = ok(ps);
                let perrs = counts(ps, "errors");
                let psw = counts(ps, "sorry_warnings");
                let psorries = picked.sorries.unwrap_or(999);

                // 1) Headline (one line, highest signal).
                let elapsed_ms = prof_t0.elapsed().as_millis() as u64;
                eprintln!(
                    "[tree-search-nearest] picked={} ok={} errors={} sorry_warnings={} sorries={} depth={} elapsed_ms={} budget_s={} (Δsorries={}, Δsw={})",
                    pick,
                    pok,
                    perrs,
                    psw,
                    psorries,
                    picked.depth,
                    elapsed_ms,
                    total_timeout_s,
                    (psorries as i64) - (0i64 + (plc::locate_sorries_in_text(&original_text, 500, 1).unwrap_or_default().len() as i64)),
                    (psw as i64) - (b_sw as i64),
                );
                if let Some(dn) = focus_decl_name.as_deref() {
                    eprintln!(
                        "[tree-search-nearest] focus decl={dn} line={}",
                        focus_line_1.unwrap_or(0)
                    );
                }
                if bailed_total_timeout {
                    eprintln!(
                        "[tree-search-nearest] bailout total_timeout=true remaining_ms={}",
                        remaining_ms(run_deadline)
                    );
                }

                if log_level >= 2 {
                    let oracle_n = events_by_kind.get("oracle_call").copied().unwrap_or(0);
                    let verify_node_n = events_by_kind.get("verify_node").copied().unwrap_or(0);
                    let bailout_n = events_by_kind
                        .get("bailout_total_timeout")
                        .copied()
                        .unwrap_or(0);
                    let seed_call_n = events_by_kind.get("oracle_seed_call").copied().unwrap_or(0);
                    let seed_result_n = events_by_kind
                        .get("oracle_seed_result")
                        .copied()
                        .unwrap_or(0);
                    let seed_skipped_n = events_by_kind
                        .get("oracle_seed_skipped")
                        .copied()
                        .unwrap_or(0);
                    let interesting = bailed_total_timeout
                        || bailout_n > 0
                        || oracle_n > 0
                        || verify_node_n > 0
                        || b_errs > 0
                        || b_warns > 0
                        || b_sw > 0
                        || picked.depth > 0;

                    if interesting {
                        // Prefer a compact view: omit low-signal boilerplate events unless we're in a total-timeout
                        // situation where they explain why no work happened.
                        let include_seed_skipped =
                            bailed_total_timeout || seed_call_n > 0 || seed_result_n > 0;
                        let is_noise_kind = |k: &str| -> bool {
                            k == "start"
                                || k == "budgets"
                                || (!include_seed_skipped && k == "oracle_seed_skipped")
                        };
                        let filtered: Vec<&serde_json::Value> = events_tail
                            .iter()
                            .filter(|ev| {
                                let k = ev.get("kind").and_then(|v| v.as_str()).unwrap_or("");
                                !is_noise_kind(k)
                            })
                            .collect();
                        let list = if filtered.is_empty() {
                            events_tail.iter().collect()
                        } else {
                            filtered
                        };

                        eprintln!(
                            "[tree-search-nearest] events bailout={} oracle_call={} verify_node={} seed_call={} seed_skipped={}",
                            bailout_n,
                            oracle_n,
                            verify_node_n,
                            seed_call_n,
                            seed_skipped_n
                        );

                        let max_events = 16usize;
                        let shown = list.len().min(max_events);
                        eprintln!("[tree-search-nearest] timeline (last {shown} events)");
                        for ev in list.iter().rev().take(shown).rev() {
                            let kind = ev.get("kind").and_then(|v| v.as_str()).unwrap_or("?");
                            let t_ms = ev.get("t_ms").and_then(|v| v.as_u64()).unwrap_or(0);
                            let detail = match kind {
                                "baseline_verify" => ev
                                    .get("skipped")
                                    .and_then(|v| v.as_bool())
                                    .filter(|b| *b)
                                    .map(|_| "skipped=true".to_string())
                                    .or_else(|| {
                                        ev.get("ms")
                                            .and_then(|v| v.as_u64())
                                            .map(|ms| format!("ms={ms}"))
                                    })
                                    .unwrap_or_default(),
                                "focus" => {
                                    let line = ev.get("line").and_then(|v| v.as_u64()).unwrap_or(0);
                                    format!("line={line}")
                                }
                                "oracle_result" => {
                                    let ok =
                                        ev.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
                                    let n = ev
                                        .get("suggestions_n")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0);
                                    let ms = ev.get("ms").and_then(|v| v.as_u64()).unwrap_or(0);
                                    format!("ok={ok} n={n} ms={ms}")
                                }
                                "verify_node" => {
                                    let ok =
                                        ev.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
                                    let s = ev.get("sorries").and_then(|v| v.as_u64()).unwrap_or(0);
                                    let cache =
                                        ev.get("cache").and_then(|v| v.as_str()).unwrap_or("?");
                                    let ms_opt = ev.get("ms").and_then(|v| v.as_u64());
                                    if let Some(ms) = ms_opt {
                                        format!("ok={ok} sorries={s} ms={ms} cache={cache}")
                                    } else {
                                        format!("ok={ok} sorries={s} cache={cache}")
                                    }
                                }
                                "end" => {
                                    let rem = ev
                                        .get("remaining_ms")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0);
                                    format!("remaining_ms={rem}")
                                }
                                _ => String::new(),
                            };
                            if detail.is_empty() {
                                eprintln!("  t={t_ms:>6} kind={kind}");
                            } else {
                                eprintln!("  t={t_ms:>6} kind={kind} {detail}");
                            }
                        }
                    }
                }

                // 2) Brief summary (baseline vs best vs best_progress).
                if summary_level >= 2 && log_level >= 1 {
                    let bp = best_progress.verify_summary.as_ref().unwrap();
                    eprintln!(
                        "[tree-search-nearest] baseline ok={} e={} sw={} w={} | best ok={} e={} sw={} w={} sorries={} | best_progress ok={} e={} sw={} w={} sorries={}",
                        b_ok,
                        b_errs,
                        b_sw,
                        b_warns,
                        best_ok2,
                        best_errs,
                        best_sw,
                        best_warns,
                        best_sorries,
                        ok(bp),
                        counts(bp, "errors"),
                        counts(bp, "sorry_warnings"),
                        counts(bp, "warnings"),
                        best_progress.sorries.unwrap_or(999)
                    );
                }

                // 3) Detail: small “top nodes” table (bounded) + failure-mode hint.
                if summary_level >= 3 && include_trace && log_level >= 2 {
                    let mut xs = all.clone();
                    xs.sort_by(|a, b| {
                        let sa = a.verify_summary.as_ref().unwrap();
                        let sb = b.verify_summary.as_ref().unwrap();
                        let ka = verify_score_key(
                            sa,
                            a.sorries.unwrap_or(999),
                            a.conservative_sorries.unwrap_or(999),
                        );
                        let kb = verify_score_key(
                            sb,
                            b.sorries.unwrap_or(999),
                            b.conservative_sorries.unwrap_or(999),
                        );
                        ka.cmp(&kb).then_with(|| a.id.cmp(&b.id))
                    });
                    eprintln!(
                        "[tree-search-nearest] trace nodes={} (showing top 6)",
                        all.len()
                    );
                    for n in xs.into_iter().take(6) {
                        let s = n.verify_summary.as_ref().unwrap();
                        let first = s.get("first_error").and_then(|v| v.as_str()).unwrap_or("");
                        let mode = classify_failure_mode(Some(first));
                        let repl = n
                            .last_replacement
                            .as_deref()
                            .unwrap_or("")
                            .replace('\n', " ");
                        let repl = if repl.chars().count() > 72 {
                            repl.chars().take(72).collect::<String>() + "…"
                        } else {
                            repl
                        };
                        let first = first.replace('\n', " ");
                        let first = if first.chars().count() > 80 {
                            first.chars().take(80).collect::<String>() + "…"
                        } else {
                            first
                        };
                        eprintln!(
                            "  - id={} d={} ok={} e={} sw={} sorries={} mode={} repl={} err={}",
                            n.id,
                            n.depth,
                            ok(s),
                            counts(s, "errors"),
                            counts(s, "sorry_warnings"),
                            n.sorries.unwrap_or(999),
                            mode,
                            repl,
                            first
                        );
                    }
                }

                // Aux evidence.
                if write {
                    eprintln!("[tree-search-nearest] wrote {}", abs.display());
                }
            }

            let mut written_file: Option<String> = None;
            if write || write_to.is_some() {
                let target: std::path::PathBuf = if let Some(p) = write_to.as_ref() {
                    if p.is_absolute() {
                        p.clone()
                    } else {
                        repo_root.join(p)
                    }
                } else {
                    abs.clone()
                };
                if let Some(parent) = target.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| format!("failed to create dir {}: {}", parent.display(), e))?;
                }
                std::fs::write(&target, picked.text.as_bytes())
                    .map_err(|e| format!("write {}: {e}", target.display()))?;
                written_file = Some(target.display().to_string());
            }

            let mut diff_written: Option<String> = None;
            let mut diff_unified: serde_json::Value = serde_json::Value::Null;
            if include_diff || output_diff.is_some() {
                let (d, truncated) =
                    unified_diff_bounded(&original_text, &picked.text, diff_context, 120_000);
                if let Some(p) = output_diff.as_ref() {
                    if let Some(parent) = p.parent() {
                        std::fs::create_dir_all(parent).map_err(|e| {
                            format!("failed to create dir {}: {}", parent.display(), e)
                        })?;
                    }
                    std::fs::write(p, d.as_bytes())
                        .map_err(|e| format!("write diff {}: {e}", p.display()))?;
                    diff_written = Some(p.display().to_string());
                }
                if include_diff {
                    diff_unified = json!({
                        "unified": d,
                        "context": diff_context,
                        "truncated": truncated
                    });
                }
            }

            let trace: serde_json::Value = if include_trace {
                let xs: Vec<serde_json::Value> = all
                    .iter()
                    .map(|n| {
                        json!({
                            "id": n.id,
                            "parent": n.parent_id,
                            "depth": n.depth,
                            "last_region": n.last_region.map(|(a,b)| json!({"start_line": a, "end_line": b})),
                            "last_replacement": n.last_replacement,
                            "sorries": n.sorries,
                            "conservative_sorries": n.conservative_sorries,
                            "verify": {
                                "summary": n.verify_summary,
                                "raw": if include_raw_verify { n.verify_raw.clone().unwrap_or(serde_json::Value::Null) } else { serde_json::Value::Null }
                            }
                        })
                    })
                    .collect();
                serde_json::Value::Array(xs)
            } else {
                serde_json::Value::Null
            };

            // Optional human-friendly Markdown report.
            let mut report_md_preview: Option<String> = None;
            let report_md_written: Option<String> = if let Some(p) = report_md.as_ref() {
                let baseline_sorries = plc::locate_sorries_in_text(&original_text, 500, 1)
                    .unwrap_or_default()
                    .len();
                let ps = picked.verify_summary.as_ref().unwrap();
                let first_err = ps.get("first_error").and_then(|v| v.as_str()).unwrap_or("");
                let mode = classify_failure_mode(Some(first_err));
                let mut md = String::new();
                md.push_str("## proofpatch tree-search-nearest report\n\n");
                md.push_str(&format!("- file: `{}`\n", file));
                md.push_str(&format!("- candidates: `{}`\n", candidates_mode));
                md.push_str(&format!("- picked: `{}` (depth {})\n", pick, picked.depth));
                md.push_str(&format!(
                    "- picked ok/errors/sw/sorries: `{}/{}/{}/{}`\n",
                    ps.get("ok").and_then(|v| v.as_bool()).unwrap_or(false),
                    ps.get("counts")
                        .and_then(|c| c.get("errors"))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0),
                    ps.get("counts")
                        .and_then(|c| c.get("sorry_warnings"))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0),
                    picked.sorries.unwrap_or(999),
                ));
                md.push_str(&format!("- baseline sorries: `{}`\n", baseline_sorries));
                if let Some(dn) = focus_decl_name.as_deref() {
                    md.push_str(&format!(
                        "- focus decl: `{}` (line {})\n",
                        dn,
                        focus_line_1.unwrap_or(0)
                    ));
                }
                md.push_str(&format!("- focus source: `{}`\n", focus_source));
                if focus_source == "focus_decl_not_found_fallback" {
                    if let Some(req) = focus_requested_decl.as_deref() {
                        md.push_str(&format!("- focus requested decl: `{}`\n", req));
                    }
                    if !focus_available_decls.is_empty() {
                        md.push_str("- focus available decls (sample):\n");
                        for d in focus_available_decls.iter().take(12) {
                            md.push_str(&format!("  - `{}`\n", d));
                        }
                    }
                }
                md.push_str("\n### Primary failure mode\n\n");
                md.push_str(&format!("- mode: `{}`\n", mode));
                if !first_err.trim().is_empty() {
                    md.push_str("\n```\n");
                    md.push_str(first_err.trim());
                    md.push_str("\n```\n");
                }
                if include_trace {
                    md.push_str("\n### Top nodes (first 6)\n\n");
                    if let Some(arr) = trace.as_array() {
                        for n in arr.iter().take(6) {
                            let id = n.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
                            let depth = n.get("depth").and_then(|v| v.as_u64()).unwrap_or(0);
                            let sorries = n.get("sorries").and_then(|v| v.as_u64()).unwrap_or(0);
                            let okv = n
                                .get("verify")
                                .and_then(|v| v.get("summary"))
                                .and_then(|v| v.get("ok"))
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false);
                            let errs = n
                                .get("verify")
                                .and_then(|v| v.get("summary"))
                                .and_then(|v| v.get("counts"))
                                .and_then(|v| v.get("errors"))
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                            let sw = n
                                .get("verify")
                                .and_then(|v| v.get("summary"))
                                .and_then(|v| v.get("counts"))
                                .and_then(|v| v.get("sorry_warnings"))
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                            let repl = n
                                .get("last_replacement")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .replace('\n', " ");
                            md.push_str(&format!(
                                "- id={} depth={} ok={} e={} sw={} sorries={} repl=`{}`\n",
                                id, depth, okv, errs, sw, sorries, repl
                            ));
                        }
                    }
                }
                md.push_str("\n### Next actions\n\n");
                md.push_str("- Re-run with `--summary-level 3 --include-trace` for a compact top-nodes view.\n");
                md.push_str(
                    "- Re-run with `--report-md <path>` to persist this report alongside JSON.\n",
                );
                md.push_str("- If using `lean-try`, consider increasing `--lean-oracle-max-calls` and `--max-nodes`.\n");
                md.push_str("- If the search is bouncing between holes, increase `--goal-first-k` (default 3 for lean-try).\n");
                md.push_str("- If you want to prioritize actually closing goals, try `--fill-mode strict` or `--fill-mode hybrid`.\n");
                md.push_str("- If you want exploration, enable `--escalate-llm` (and configure provider).\n");

                // Keep a small inline preview in the JSON output. This makes the report usable even when
                // the report file path is inconvenient to open in an IDE (e.g. ignored/generated dirs).
                let max_preview_chars = 8000usize;
                let total_chars = md.chars().count();
                let (kept, truncated) = if total_chars > max_preview_chars {
                    (md.chars().take(max_preview_chars).collect::<String>(), true)
                } else {
                    (md.clone(), false)
                };
                report_md_preview = Some(if truncated {
                    format!("{kept}\n\n[proofpatch: report_md preview truncated]\n")
                } else {
                    kept
                });

                if let Some(parent) = p.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| format!("failed to create dir {}: {}", parent.display(), e))?;
                }
                std::fs::write(p, md.as_bytes())
                    .map_err(|e| format!("write report {}: {e}", p.display()))?;
                Some(p.display().to_string())
            } else {
                None
            };

            let events_jsonl_effective: Option<String> =
                events_jsonl.as_ref().map(|p| p.display().to_string());
            let events_jsonl_written: Option<String> = if let Some(p) = events_jsonl.as_ref() {
                if let Some(parent) = p.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let mut buf = String::new();
                for ev in &events_all {
                    buf.push_str(&ev.to_string());
                    buf.push('\n');
                }
                if std::fs::write(p, buf.as_bytes()).is_ok() {
                    Some(p.display().to_string())
                } else {
                    None
                }
            } else {
                None
            };

            let elapsed_ms_final = prof_t0.elapsed().as_millis() as u64;
            let remaining_ms_final = remaining_ms(run_deadline);
            let verify_node_total = events_by_kind.get("verify_node").copied().unwrap_or(0);
            let oracle_call_total = events_by_kind.get("oracle_call").copied().unwrap_or(0);
            let goal_dump_total = events_by_kind.get("goal_dump").copied().unwrap_or(0);

            let mut out = json!({
                "repo_root": repo_root.display().to_string(),
                "file": file,
                "written_file": written_file,
                "result_kind": "search",
                "note": if hard_focus_stuck {
                    serde_json::Value::String("Hard focus prevented further expansion: no remaining `sorry` in focus decl on some nodes (and drift is disabled).".to_string())
                } else {
                    serde_json::Value::Null
                },
                "write_mode": if write { "inplace" } else if write_to.is_some() { "to_path" } else { "none" },
                "diff": diff_unified,
                "diff_written": diff_written,
                "artifacts": {
                    "events_jsonl": {
                        "requested": events_jsonl_requested.as_ref().map(|p| p.display().to_string()),
                        "effective": events_jsonl_effective,
                        "written": events_jsonl_written
                    },
                    "report_md": {
                        "requested": report_md_requested.as_ref().map(|p| p.display().to_string()),
                        "written": report_md_written,
                        "preview": report_md_preview
                    },
                    "diff_output": {
                        "requested": output_diff_requested.as_ref().map(|p| p.display().to_string()),
                        "effective": output_diff.as_ref().map(|p| p.display().to_string()),
                        "written": diff_written
                    }
                },
                "events": {
                    "counts_by_kind": events_by_kind,
                    "limits": {
                        "keep": events_keep,
                        "all_keep": events_all_keep,
                    },
                    "stats": {
                        "elapsed_ms": elapsed_ms_final,
                        "remaining_ms": remaining_ms_final,
                        "baseline_verify": {
                            "ms": if baseline_skipped { serde_json::Value::Null } else { json!(baseline_ms) },
                            "skipped": baseline_skipped,
                            "ok": baseline_summary.get("ok").cloned().unwrap_or(serde_json::Value::Null),
                            "counts": baseline_summary.get("counts").cloned().unwrap_or(serde_json::Value::Null),
                        },
                        "timers_ms": {
                            "verify_baseline_ms": prof_verify_baseline_ms,
                            "verify_nodes_ms": prof_verify_nodes_ms,
                            "goal_dump_ms": prof_goal_dump_ms,
                            "lean_suggest_ms": prof_lean_suggest_ms,
                            "locate_sorries_ms": prof_locate_sorries_ms,
                            "conservative_sorries_ms": prof_conservative_sorries_ms,
                            "patch_ms": prof_patch_ms,
                        },
                        "caches": {
                            "disk_eval": {
                                "hits": disk_cache_eval_hits,
                                "misses": disk_cache_eval_misses
                            },
                            "oracle_suggestions": {
                                "cache_hits": lean_oracle_cache_hits,
                                "cache_misses": lean_oracle_cache_misses,
                                "state_cache_hits": lean_state_cache_hits,
                                "state_cache_misses": lean_state_cache_misses,
                                "goal_dedup_skips": lean_oracle_goal_dedup_skips,
                            },
                            "goal_dump": {
                                "cache_hits": goal_dump_cache_hits,
                                "cache_misses": goal_dump_cache_misses,
                                "hyps_cache_hits": goal_dump_hyps_cache_hits,
                                "hyps_cache_misses": goal_dump_hyps_cache_misses,
                            }
                        },
                        "calls": {
                            "verify_baseline": prof_verify_baseline_calls,
                            "verify_nodes_verified": prof_verify_nodes_calls,
                            "verify_nodes_total": verify_node_total,
                            "oracle_call_total": oracle_call_total,
                            "goal_dump_total": goal_dump_total,
                        }
                    },
                    "tail": events_tail,
                },
                "config": {
                    "timeout_s": timeout_s,
                    "total_timeout_s": total_timeout_s,
                    "goal_dump_timeout_s": goal_dump_timeout_s,
                    "oracle_timeout_s": oracle_timeout_s,
                    "log_level": log_level,
                    "beam": beam,
                    "max_nodes": max_nodes,
                    "depth": depth,
                    "candidates": candidates_mode,
                    "candidates_count": candidates.len(),
                    "decision_effects": decision_effects,
                    "lean_oracle_per_node": lean_oracle_per_node,
                    "lean_oracle_max_calls": lean_oracle_max_calls,
                    "rollout_k": rollout_k,
                    "dedup_goal_expansions": dedup_goal_expansions,
                    "goal_first_k": goal_first_k,
                    "goal_meta_penalty": goal_meta_penalty,
                    "depth_bonus": depth_bonus,
                    "fill_mode": fill_mode,
                    "profile": profile,
                    "llm_summary": llm_summary,
                    "llm_summary_timeout_s": llm_summary_timeout_s,
                    "llm_planner": llm_planner,
                    "smt_precheck": smt_precheck,
                    "smt_depth": smt_depth,
                    "smt_timeout_ms": smt_timeout_ms,
                    "smt_seed": smt_seed,
                    "escalate_llm": escalate_llm,
                    "allow_sorry_candidates": allow_sorry_candidates,
                    "include_trace": include_trace,
                    "pick": pick,
                    "research_notes": research_notes,
                    "research_preset": research_preset,
                    "research_top_k": research_top_k,
                    "include_diff": include_diff,
                    "diff_context": diff_context,
                    "max_candidates_per_node": max_candidates_per_node,
                    "verify_k": verify_k,
                    "effective_max_candidates_per_node": if effective_max_candidates_per_node == usize::MAX { serde_json::Value::Null } else { json!(effective_max_candidates_per_node) },
                    "effective_verify_k": if effective_verify_k == usize::MAX { serde_json::Value::Null } else { json!(effective_verify_k) },
                    "cache": {
                        "enabled": cache_dir.is_some(),
                        "dir": cache_dir.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "".to_string()),
                    },
                },
                "bailouts": {
                    "total_timeout": bailed_total_timeout,
                    "focus_decl_hard_stuck": hard_focus_stuck,
                },
                "oracle": {
                    "calls": lean_oracle_calls,
                    "cache_hits": lean_oracle_cache_hits,
                    "cache_misses": lean_oracle_cache_misses,
                    "cache_size": lean_oracle_cache.len(),
                    "goal_dedup_skips": lean_oracle_goal_dedup_skips,
                    "state_cache": {
                        "hits": lean_state_cache_hits,
                        "misses": lean_state_cache_misses,
                        "size": lean_state_candidates_cache.len()
                    },
                    "goal_dump": {
                        "calls": goal_dump_calls,
                        "cache_hits": goal_dump_cache_hits,
                        "cache_misses": goal_dump_cache_misses,
                        "cache_size": goal_dump_cache.len()
                    },
                    "smt": {
                        "cache_hits": smt_cache_hits,
                        "cache_misses": smt_cache_misses,
                        "entails_attempts": smt_entails_attempts,
                        "entails_escalations": smt_entails_escalations,
                        "entails_trace": smt_entails_trace,
                        "errors": smt_errors,
                        "last_error": smt_last_error,
                        "note": if smt_dump && smt_dumps_written == 0 && !goal_dump {
                            json!("SMT dumping typically needs goal dumps; re-run with `--goal-dump` (and `--smt-precheck`) to populate hypotheses/targets.")
                        } else {
                            serde_json::Value::Null
                        },
                        "proofs": {
                            "enabled": { "value": smt_proof, "source": smt_proof_source },
                            "max_chars": smt_proof_max_chars,
                            "attempts": smt_proof_attempts,
                            "captured": smt_proofs_captured,
                            "last_error": smt_proof_last_error,
                            "last": smt_proof_last,
                            "note": if smt_proof && smt_proof_attempts == 0 {
                                if smt_proof_dump && goal_dump_v.is_none() {
                                    json!("SMT proof dumping typically needs goal dumps; re-run with `--goal-dump` (and usually `--smt-precheck`) so `proofpatch` can see hypotheses/targets.")
                                } else {
                                    json!("UNSAT proof objects are only available when an entailment check returns `entails=true` (i.e., the SMT query finds UNSAT for `hyps ∧ ¬target`).")
                                }
                            } else if smt_proof && smt_proof_attempts > 0 && smt_proofs_captured == 0 && smt_proof_last_error.is_none() {
                                json!("No proof was captured; this can happen if the solver reports `unknown`, does not support `(get-proof)`, or returns `unsupported` for this query shape.")
                            } else {
                                serde_json::Value::Null
                            },
                            "dump": {
                                "enabled": smt_proof_dump,
                                "dir": smt_proof_dump_dir_opt
                                    .as_ref()
                                    .map(|p| p.display().to_string())
                                    .unwrap_or_else(|| "".to_string()),
                                "max_chars": smt_proof_dump_max_chars,
                                "attempts": smt_proof_dump_attempts,
                                "written": smt_proof_dump_written,
                                "skipped_too_large": smt_proof_dump_skipped_too_large,
                                "last_error": smt_proof_dump_last_error,
                                "paths": smt_proof_dump_paths,
                            },
                        },
                        "dumps_written": smt_dumps_written,
                        "dump_paths": smt_dump_paths,
                        "dump_last": {
                            "path": smt_dump_last_path,
                            "chars": smt_dump_last_chars,
                            "preview": smt_dump_last_preview,
                        },
                        "goal_dump_hyps_cache_hits": goal_dump_hyps_cache_hits,
                        "goal_dump_hyps_cache_misses": goal_dump_hyps_cache_misses,
                        "solver": plc::smt_lia::smt_solver_probe(),
                        "reuse": smt_reuse
                            .as_ref()
                            .map(|s| s.stats())
                            .unwrap_or(serde_json::Value::Null),
                    }
                },
                "baseline_verify": { "summary": baseline_summary },
                "research_context": research_context,
                "goal_dump": goal_dump_v,
                "lean_suggestions": lean_suggest_v,
                "llm": {
                    "initial": llm_meta_initial,
                    "escalate": {
                        "attempts": llm_escalate_attempts,
                        "successes": llm_escalate_successes,
                        "last_error": llm_escalate_last_error
                    },
                    "summary": serde_json::Value::Null
                },
                "best": {
                    "id": best.id,
                    "depth": best.depth,
                    "parent": best.parent_id,
                    "last_region": best.last_region.map(|(a,b)| json!({"start_line": a, "end_line": b})),
                    "last_replacement": best.last_replacement,
                    "smt_support_used": best
                        .last_replacement
                        .as_ref()
                        .map(|s| s.contains("proofpatch:smt_support"))
                        .unwrap_or(false),
                    "sorries": best.sorries,
                    "conservative_sorries": best.conservative_sorries,
                    "focus_goal_sig": best.focus_goal_sig,
                    // Canonical, stable field for SMT provenance/artifacts.
                    "smt_evidence": best.smt_hint.clone(),
                    // Back-compat alias.
                    "smt_hint": best.smt_hint,
                    "rank_hint": best.rank_hint,
                    "verify": {
                        "summary": best.verify_summary,
                        "raw": if include_raw_verify { best.verify_raw.clone().unwrap_or(serde_json::Value::Null) } else { serde_json::Value::Null }
                    }
                },
                "best_progress": {
                    "id": best_progress.id,
                    "depth": best_progress.depth,
                    "parent": best_progress.parent_id,
                    "last_region": best_progress.last_region.map(|(a,b)| json!({"start_line": a, "end_line": b})),
                    "last_replacement": best_progress.last_replacement,
                    "smt_support_used": best_progress
                        .last_replacement
                        .as_ref()
                        .map(|s| s.contains("proofpatch:smt_support"))
                        .unwrap_or(false),
                    "sorries": best_progress.sorries,
                    "conservative_sorries": best_progress.conservative_sorries,
                    "focus_goal_sig": best_progress.focus_goal_sig,
                    "smt_evidence": best_progress.smt_hint.clone(),
                    "smt_hint": best_progress.smt_hint,
                    "rank_hint": best_progress.rank_hint,
                    "verify": {
                        "summary": best_progress.verify_summary,
                        "raw": if include_raw_verify { best_progress.verify_raw.clone().unwrap_or(serde_json::Value::Null) } else { serde_json::Value::Null }
                    }
                },
                "best_ok": {
                    "id": best_ok.id,
                    "depth": best_ok.depth,
                    "parent": best_ok.parent_id,
                    "last_region": best_ok.last_region.map(|(a,b)| json!({"start_line": a, "end_line": b})),
                    "last_replacement": best_ok.last_replacement,
                    "smt_support_used": best_ok
                        .last_replacement
                        .as_ref()
                        .map(|s| s.contains("proofpatch:smt_support"))
                        .unwrap_or(false),
                    "sorries": best_ok.sorries,
                    "conservative_sorries": best_ok.conservative_sorries,
                    "focus_goal_sig": best_ok.focus_goal_sig,
                    "smt_evidence": best_ok.smt_hint.clone(),
                    "smt_hint": best_ok.smt_hint,
                    "rank_hint": best_ok.rank_hint,
                    "verify": {
                        "summary": best_ok.verify_summary,
                        "raw": if include_raw_verify { best_ok.verify_raw.clone().unwrap_or(serde_json::Value::Null) } else { serde_json::Value::Null }
                    }
                },
                "picked": {
                    "id": picked.id,
                    "depth": picked.depth,
                    "parent": picked.parent_id,
                    "last_region": picked.last_region.map(|(a,b)| json!({"start_line": a, "end_line": b})),
                    "last_replacement": picked.last_replacement,
                    "smt_support_used": picked
                        .last_replacement
                        .as_ref()
                        .map(|s| s.contains("proofpatch:smt_support"))
                        .unwrap_or(false),
                    "sorries": picked.sorries,
                    "conservative_sorries": picked.conservative_sorries,
                    "focus_goal_sig": picked.focus_goal_sig,
                    "smt_evidence": picked.smt_hint.clone(),
                    "smt_hint": picked.smt_hint,
                    "rank_hint": picked.rank_hint,
                    "verify": {
                        "summary": picked.verify_summary,
                        "raw": if include_raw_verify { picked.verify_raw.clone().unwrap_or(serde_json::Value::Null) } else { serde_json::Value::Null }
                    }
                },
                // Extra “artifact pointers” so consumers don't have to parse nested JSON to
                // recover the selected patch text. When caching is enabled, we also write
                // the replacement text to a stable path under `.generated/proofpatch-cache/`.
                "replacement_artifacts": {
                    "picked": {
                        "region": picked.last_region.map(|(a,b)| json!({"start_line": a, "end_line": b})),
                        "preview": picked.last_replacement.as_ref().map(|s| truncate_str(s, 400)),
                        "chars": picked.last_replacement.as_ref().map(|s| s.chars().count()),
                        "hash64": picked.last_replacement.as_ref().map(|s| hash_text(s)),
                        "path": picked.last_replacement.as_ref().and_then(|s| {
                            cache_dir.as_ref().map(|root| {
                                let h = hash_text(s);
                                let rel = format!("tree_search/replacements/picked_{h}.lean");
                                durable_atomic_write(root, &rel, s.as_bytes());
                                root.join(rel).display().to_string()
                            })
                        })
                    },
                    "best_ok": {
                        "region": best_ok.last_region.map(|(a,b)| json!({"start_line": a, "end_line": b})),
                        "preview": best_ok.last_replacement.as_ref().map(|s| truncate_str(s, 400)),
                        "chars": best_ok.last_replacement.as_ref().map(|s| s.chars().count()),
                        "hash64": best_ok.last_replacement.as_ref().map(|s| hash_text(s)),
                        "path": best_ok.last_replacement.as_ref().and_then(|s| {
                            cache_dir.as_ref().map(|root| {
                                let h = hash_text(s);
                                let rel = format!("tree_search/replacements/best_ok_{h}.lean");
                                durable_atomic_write(root, &rel, s.as_bytes());
                                root.join(rel).display().to_string()
                            })
                        })
                    },
                    "best_progress": {
                        "region": best_progress.last_region.map(|(a,b)| json!({"start_line": a, "end_line": b})),
                        "preview": best_progress.last_replacement.as_ref().map(|s| truncate_str(s, 400)),
                        "chars": best_progress.last_replacement.as_ref().map(|s| s.chars().count()),
                        "hash64": best_progress.last_replacement.as_ref().map(|s| hash_text(s)),
                        "path": best_progress.last_replacement.as_ref().and_then(|s| {
                            cache_dir.as_ref().map(|root| {
                                let h = hash_text(s);
                                let rel = format!("tree_search/replacements/best_progress_{h}.lean");
                                durable_atomic_write(root, &rel, s.as_bytes());
                                root.join(rel).display().to_string()
                            })
                        })
                    }
                },
                // A compact summary that is easy to `jq` / grep.
                "human_summary": {
                    "picked": pick,
                    "focus_decl": focus_decl_name,
                    "focus_line": focus_line_1,
                    "focus_decl_hard": focus_decl_hard,
                    "baseline": baseline_summary,
                    "picked_verify": picked.verify_summary
                },
                "focus": {
                    "source": focus_source,
                    "requested_decl": focus_requested_decl,
                    "available_decls": focus_available_decls,
                    "primary_sorry": focus_sorry.as_ref().map(|s| json!({
                        "token": s.token,
                        "decl_kind": s.decl_kind,
                        "decl_name": s.decl_name,
                        "decl_line": s.decl_line,
                        "line": s.line,
                        "col": s.col,
                        "region_start": s.region_start,
                        "region_end": s.region_end,
                        "line_text": truncate_str(&s.line_text, 200),
                        "excerpt": truncate_str(&s.excerpt, 800),
                    }))
                },
                "trace": trace
            });

            if profile {
                let total_ms = prof_t0.elapsed().as_millis() as u64;
                #[cfg(feature = "planner")]
                let (planner_ms_u64, planner_hits_u64, planner_misses_u64) =
                    (prof_planner_ms, planner_cache_hits, planner_cache_misses);
                #[cfg(not(feature = "planner"))]
                let (planner_ms_u64, planner_hits_u64, planner_misses_u64) = (0u64, 0u64, 0u64);
                let (smt_ms_u64, smt_hits_u64, smt_misses_u64) =
                    (prof_smt_ms, smt_cache_hits, smt_cache_misses);
                let accounted = prof_verify_baseline_ms
                    .saturating_add(prof_verify_nodes_ms)
                    .saturating_add(prof_goal_dump_ms)
                    .saturating_add(prof_lean_suggest_ms)
                    .saturating_add(planner_ms_u64)
                    .saturating_add(smt_ms_u64)
                    .saturating_add(prof_locate_sorries_ms)
                    .saturating_add(prof_conservative_sorries_ms)
                    .saturating_add(prof_patch_ms);
                let misc_ms = total_ms.saturating_sub(accounted);
                if let Some(obj) = out.as_object_mut() {
                    obj.insert(
                        "profile".to_string(),
                        json!({
                            "total_ms": total_ms,
                            "verify_baseline_ms": prof_verify_baseline_ms,
                            "verify_nodes_ms": prof_verify_nodes_ms,
                            "goal_dump_ms": prof_goal_dump_ms,
                            "lean_suggest_ms": prof_lean_suggest_ms,
                            "planner_ms": planner_ms_u64,
                            "smt_ms": smt_ms_u64,
                            "locate_sorries_ms": prof_locate_sorries_ms,
                            "conservative_sorries_ms": prof_conservative_sorries_ms,
                            "patch_ms": prof_patch_ms,
                            "verify_baseline_calls": prof_verify_baseline_calls,
                            "verify_nodes_calls": prof_verify_nodes_calls,
                            "candidates_considered": prof_candidates_considered,
                            "candidates_verified": prof_candidates_verified,
                            "disk_cache_eval_hits": disk_cache_eval_hits,
                            "disk_cache_eval_misses": disk_cache_eval_misses,
                            "planner_cache_hits": planner_hits_u64,
                            "planner_cache_misses": planner_misses_u64,
                            "smt_cache_hits": smt_hits_u64,
                            "smt_cache_misses": smt_misses_u64,
                            "misc_ms": misc_ms,
                        }),
                    );
                }
            }

            // Optional LLM-generated multi-scale summary (added to machine JSON; human printing remains separate).
            if llm_summary {
                let evidence = json!({
                    "repo_root": repo_root.display().to_string(),
                    "file": file,
                    "config": out.get("config").cloned().unwrap_or(serde_json::Value::Null),
                    "oracle": out.get("oracle").cloned().unwrap_or(serde_json::Value::Null),
                    "baseline_verify": out.get("baseline_verify").cloned().unwrap_or(serde_json::Value::Null),
                    "best": out.get("best").cloned().unwrap_or(serde_json::Value::Null),
                    "best_progress": out.get("best_progress").cloned().unwrap_or(serde_json::Value::Null),
                    "best_ok": out.get("best_ok").cloned().unwrap_or(serde_json::Value::Null),
                    "picked": out.get("picked").cloned().unwrap_or(serde_json::Value::Null),
                    "trace_top": if include_trace {
                        out.get("trace").and_then(|v| v.as_array()).map(|a| {
                            serde_json::Value::Array(a.iter().take(6).cloned().collect())
                        }).unwrap_or(serde_json::Value::Null)
                    } else { serde_json::Value::Null },
                    "first_error": {
                        "baseline": baseline_summary.get("first_error").cloned().unwrap_or(serde_json::Value::Null),
                        "picked": picked.verify_summary.as_ref().and_then(|s| s.get("first_error")).cloned().unwrap_or(serde_json::Value::Null)
                    }
                });

                let system = r#"You are a theorem-proving tooling analyst.
Given a JSON evidence packet for a Lean proof search run, produce an LLM-generated summary with three scales.

Return STRICT JSON (no markdown fences) with keys:
- short: one sentence (<= 180 chars)
- medium: 3-5 bullet strings
- long: 2-3 short paragraphs (as a single string)
- next_actions: 3-6 bullet strings, concrete and tool/flag oriented
- diagnostics: object with keys: primary_failure_mode, evidence_snippet (both strings)

Constraints:
- Do not hallucinate. If evidence is missing, say so.
- Prefer actionable flags/knobs and caching/search guidance."#;

                let user = serde_json::to_string(&evidence).unwrap_or_else(|_| "{}".to_string());
                let res = rt.block_on(plc::llm::chat_completion(
                    system,
                    &user,
                    StdDuration::from_secs(llm_summary_timeout_s),
                ));
                let summary_v = match res {
                    Ok(done) => {
                        let parsed = serde_json::from_str::<serde_json::Value>(&done.content).ok();
                        let parsed_ok = parsed.is_some();
                        let heuristic = json!({
                            "short": "Search run summary unavailable from LLM; using heuristic fallback.",
                            "medium": [
                                "LLM summary failed to parse or provider error; see `llm.summary.error`.",
                                "Use the stderr headline + report-md for human summary.",
                                "If stuck, increase `--depth/--beam` or enable `--escalate-llm`."
                            ],
                            "long": "The LLM summarizer did not return valid JSON (or the provider errored). This run is still fully represented by the machine JSON plus the stderr and/or Markdown report for human consumption.",
                            "next_actions": [
                                "Re-run with `--report-md <path>` to get a human-readable report artifact.",
                                "Try `--summary-level 3 --include-trace` for a compact top-nodes view.",
                                "If using `lean-try`, try increasing `--lean-oracle-max-calls`."
                            ],
                            "diagnostics": {
                                "primary_failure_mode": "llm_summary_failed",
                                "evidence_snippet": done.content.chars().take(240).collect::<String>()
                            }
                        });
                        json!({
                            "attempted": true,
                            "ok": true,
                            "error": if parsed.is_some() { serde_json::Value::Null } else { serde_json::Value::String("llm_summary_not_json".to_string()) },
                            "raw_preview": done.content.chars().take(800).collect::<String>(),
                            "parsed": parsed.unwrap_or(heuristic),
                            "source": if parsed_ok { "llm" } else { "heuristic_fallback" },
                        })
                    }
                    Err(e) => json!({
                        "attempted": true,
                        "ok": true,
                        "error": format!("{e}"),
                        "raw_preview": serde_json::Value::Null,
                        "parsed": json!({
                            "short": "LLM summary unavailable; using heuristic fallback.",
                            "medium": [
                                "LLM provider call failed; see `llm.summary.error`.",
                                "Use `--report-md <path>` for a human-readable artifact.",
                                "If stuck, increase search bounds (`--depth`, `--beam`, `--max-nodes`) or enable `--escalate-llm`."
                            ],
                            "long": "The LLM summarizer failed due to a provider/runtime error. The run is still fully described by the machine JSON output; prefer the stderr summary and/or the Markdown report for humans.",
                            "next_actions": [
                                "Re-run with `--summary-level 3 --include-trace` to see top candidate nodes.",
                                "Re-run with `--report-md <path>` to persist a human report.",
                                "If you want LLM summaries, fix the LLM provider config and retry `--llm-summary`."
                            ],
                            "diagnostics": {
                                "primary_failure_mode": "llm_provider_error",
                                "evidence_snippet": format!("{e}")
                            }
                        }),
                        "source": "heuristic_fallback"
                    }),
                };
                if let Some(obj) = out.get_mut("llm").and_then(|v| v.as_object_mut()) {
                    obj.insert("summary".to_string(), summary_v);
                }
            }

            // Human headline to stderr for transparency (keeps stdout machine-readable).
            // This is intentionally compact and stable-ish, but not a contract surface (JSON is).
            if !quiet {
                let baseline_counts = baseline_summary
                    .get("counts")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let picked_counts = picked
                    .verify_summary
                    .as_ref()
                    .and_then(|v| v.get("counts"))
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);

                let artifacts_events = out
                    .pointer("/artifacts/events_jsonl/written")
                    .and_then(|v| v.as_str())
                    .unwrap_or("-");
                let artifacts_report = out
                    .pointer("/artifacts/report_md/written")
                    .and_then(|v| v.as_str())
                    .unwrap_or("-");
                let artifacts_diff = out
                    .pointer("/artifacts/diff_output/written")
                    .and_then(|v| v.as_str())
                    .unwrap_or("-");

                let focus_decl_s = focus_decl_name.clone().unwrap_or_else(|| "-".to_string());
                let focus_line_s = focus_line_1
                    .map(|l| l.to_string())
                    .unwrap_or_else(|| "-".to_string());

                let baseline_counts_s =
                    serde_json::to_string(&baseline_counts).unwrap_or_else(|_| "null".to_string());
                let picked_counts_s =
                    serde_json::to_string(&picked_counts).unwrap_or_else(|_| "null".to_string());

                eprintln!(
                    "tree-search-nearest: pick={} focus={}@{} source={} baseline_counts={} picked_counts={} artifacts(events_jsonl={}, report_md={}, diff_output={})",
                    pick,
                    focus_decl_s,
                    focus_line_s,
                    focus_source,
                    baseline_counts_s,
                    picked_counts_s,
                    artifacts_events,
                    artifacts_report,
                    artifacts_diff
                );
            }

            // Normalize SMT proof dump paths: we may write the same file more than once
            // (e.g., best-effort + cached entailment artifact paths). Keep the JSON stable by
            // deduping paths and aligning `written` with the unique path count.
            if let Some(dump_v) = out
                .get_mut("oracle")
                .and_then(|v| v.get_mut("smt"))
                .and_then(|v| v.get_mut("proofs"))
                .and_then(|v| v.get_mut("dump"))
            {
                if let Some(dump_obj) = dump_v.as_object_mut() {
                    let mut new_len: Option<usize> = None;
                    if let Some(arr) = dump_obj.get_mut("paths").and_then(|v| v.as_array_mut()) {
                        let mut seen: std::collections::HashSet<String> =
                            std::collections::HashSet::new();
                        let mut dedup: Vec<serde_json::Value> = Vec::new();
                        for it in arr.iter() {
                            if let Some(s) = it.as_str() {
                                if seen.insert(s.to_string()) {
                                    dedup.push(it.clone());
                                }
                            }
                        }
                        *arr = dedup;
                        new_len = Some(arr.len());
                    }
                    if let Some(n) = new_len {
                        dump_obj.insert("written".to_string(), json!(n as u64));
                    }
                }
            }

            // Optional: auto-generate a durable SMT repro bundle.
            if let Some(p) = smt_repro_dir_opt.as_ref() {
                let base = if p.is_absolute() {
                    p.clone()
                } else {
                    repo_root.join(p)
                };
                let dir = if base.join("tree_search.json").exists() {
                    let ts_ms = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_else(|_| StdDuration::from_millis(0))
                        .as_millis();
                    base.join(format!("run_{ts_ms}"))
                } else {
                    base
                };
                let _ = std::fs::create_dir_all(&dir);

                // Always write the full tree-search JSON into the bundle (so it's self-contained),
                // even if `--output-json` is not set.
                let tree_search_path = dir.join("tree_search.json");
                let _ = write_json(&tree_search_path, &out);

                let pp_dump = goal_dump_v
                    .as_ref()
                    .and_then(|gd| gd.get("pp_dump"))
                    .cloned();

                let mut smt_repro_status = json!({
                    "dir": dir.display().to_string(),
                    "tree_search_json": tree_search_path.display().to_string(),
                    "ok": false,
                    "note": "no_goal_dump",
                });

                if let Some(pp_dump) = pp_dump {
                    let smt2_path = dir.join("repro.smt2");
                    let proof_path = dir.join("repro.sexp");
                    let smt_repro_json = dir.join("smt_repro.json");

                    // Reuse the same core logic as `smt-repro` (inline, bounded).
                    let timeout_ms = smt_timeout_ms;
                    let seed = smt_seed;
                    let depth = smt_depth;
                    let proof_max_chars = smt_proof_dump_max_chars;

                    let smt2 =
                        plc::smt_lia::smt2_script_from_pp_dump(&pp_dump, timeout_ms, seed, depth);
                    let smt2_has = smt2.is_some();
                    if let Some(s) = smt2.as_ref() {
                        let _ = std::fs::write(&smt2_path, s.as_bytes());
                    }
                    let proof = plc::smt_lia::unsat_proof_from_pp_dump(
                        &pp_dump,
                        timeout_ms,
                        seed,
                        depth,
                        proof_max_chars,
                    );

                    // Write proof only when not truncated.
                    let mut proof_written = false;
                    if let Ok(Some(pf)) = proof.as_ref() {
                        let preview = pf.get("preview").and_then(|v| v.as_str()).unwrap_or("");
                        let total_chars =
                            pf.get("chars").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                        let preview_chars = preview.chars().count();
                        if total_chars <= proof_max_chars && preview_chars == total_chars {
                            let _ = std::fs::write(&proof_path, preview.as_bytes());
                            proof_written = true;
                        }
                    }
                    let smt2_text = smt2.clone().unwrap_or_else(|| "".to_string());

                    let out2 = json!({
                        "ok": true,
                        "kind": "smt_repro",
                        "input_json": tree_search_path.display().to_string(),
                        "solver": plc::smt_lia::smt_solver_probe(),
                        "params": {
                            "timeout_ms": timeout_ms,
                            "seed": seed,
                            "depth": depth,
                            "proof_max_chars": proof_max_chars,
                        },
                        "artifacts": {
                            "smt2_written": if smt2_has { serde_json::Value::String(smt2_path.display().to_string()) } else { serde_json::Value::Null },
                            "proof_written": if proof_written { serde_json::Value::String(proof_path.display().to_string()) } else { serde_json::Value::Null },
                        },
                        "smt2": smt2_text,
                        "proof": match proof {
                            Ok(pf) => pf.unwrap_or(serde_json::Value::Null),
                            Err(e) => json!({"error": truncate_str(&e, 400)}),
                        }
                    });
                    let _ = write_json(&smt_repro_json, &out2);
                    smt_repro_status = json!({
                        "dir": dir.display().to_string(),
                        "tree_search_json": tree_search_path.display().to_string(),
                        "smt_repro_json": smt_repro_json.display().to_string(),
                        "smt2": if smt2_has { smt2_path.display().to_string() } else { "".to_string() },
                        "proof": if proof_written { proof_path.display().to_string() } else { "".to_string() },
                        "ok": true,
                        "note": serde_json::Value::Null
                    });
                }

                if let Some(obj) = out.as_object_mut() {
                    if let Some(cfg) = obj.get_mut("config").and_then(|v| v.as_object_mut()) {
                        cfg.insert(
                            "smt_repro_dir".to_string(),
                            json!({
                                "value": dir.display().to_string(),
                                "source": smt_repro_dir_source
                            }),
                        );
                    }
                    if let Some(art) = obj.get_mut("artifacts").and_then(|v| v.as_object_mut()) {
                        art.insert("smt_repro".to_string(), smt_repro_status);
                    }
                }
            }

            if let Some(p) = output_json {
                write_json(&p, &out)?;
                println!(
                    "{}",
                    json!({
                        "ok": true,
                        "written": p.display().to_string(),
                        "kind": "tree_search_nearest",
                        "result_kind": out["result_kind"],
                    })
                    .to_string()
                );
            } else {
                println!("{}", out.to_string());
            }
            Ok(())
        }

        "smt-repro" => {
            let input_json = arg_value(rest, "--input-json")
                .ok_or_else(|| "missing --input-json".to_string())?;
            let timeout_ms = arg_u64(rest, "--timeout-ms")
                .unwrap_or(5_000)
                .clamp(0, 600_000);
            let seed = arg_u64(rest, "--seed").unwrap_or(0);
            let depth = arg_u64(rest, "--depth").unwrap_or(0).clamp(0, 64) as usize;
            let emit_smt2 = arg_value(rest, "--emit-smt2").map(PathBuf::from);
            let emit_proof = arg_value(rest, "--emit-proof").map(PathBuf::from);
            let bundle_dir = arg_value(rest, "--bundle-dir").map(PathBuf::from);
            let proof_max_chars = arg_u64(rest, "--proof-max-chars")
                .unwrap_or(200_000)
                .clamp(0, 5_000_000) as usize;
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);

            let input_label = input_json.clone();
            let input_text = if input_json == "-" {
                let mut s = String::new();
                std::io::stdin()
                    .read_to_string(&mut s)
                    .map_err(|e| format!("read stdin: {e}"))?;
                s
            } else {
                let p = PathBuf::from(&input_json);
                std::fs::read_to_string(&p).map_err(|e| format!("read {}: {e}", p.display()))?
            };
            let v = serde_json::from_str::<serde_json::Value>(&input_text)
                .map_err(|e| format!("json parse {input_label}: {e}"))?;

            // Accept a few shapes:
            // - a raw `pp_dump` with key `goals`
            // - a `goal_dump` record with key `pp_dump`
            // - a full `tree-search-nearest` output with key `goal_dump.pp_dump`
            let pp_dump = if v.get("goals").and_then(|x| x.as_array()).is_some() {
                v.clone()
            } else if let Some(pp) = v.get("pp_dump") {
                pp.clone()
            } else if let Some(pp) = v.get("goal_dump").and_then(|gd| gd.get("pp_dump")) {
                pp.clone()
            } else {
                return Err("input json must contain a `pp_dump` (or be a `tree-search-nearest` output with `goal_dump.pp_dump`)".to_string());
            };

            // Optional bundle dir: write a full capsule (pp_dump + outputs + manifest).
            let mut emit_smt2 = emit_smt2;
            let mut emit_proof = emit_proof;
            let mut output_json = output_json;
            if let Some(dir) = bundle_dir.as_ref() {
                std::fs::create_dir_all(dir)
                    .map_err(|e| format!("failed to create dir {}: {e}", dir.display()))?;
                if output_json.is_none() {
                    output_json = Some(dir.join("smt_repro.json"));
                }
                if emit_smt2.is_none() {
                    emit_smt2 = Some(dir.join("repro.smt2"));
                }
                if emit_proof.is_none() {
                    emit_proof = Some(dir.join("repro.sexp"));
                }
            }

            let solver_probe = plc::smt_lia::smt_solver_probe();
            let smt2 = plc::smt_lia::smt2_script_from_pp_dump(&pp_dump, timeout_ms, seed, depth);
            let proof = plc::smt_lia::unsat_proof_from_pp_dump(
                &pp_dump,
                timeout_ms,
                seed,
                depth,
                proof_max_chars,
            );

            // Optional artifacts.
            let mut smt2_written: Option<String> = None;
            if let (Some(path), Some(s)) = (emit_smt2.as_ref(), smt2.as_ref()) {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| format!("failed to create dir {}: {e}", parent.display()))?;
                }
                std::fs::write(path, s.as_bytes())
                    .map_err(|e| format!("failed to write {}: {e}", path.display()))?;
                smt2_written = Some(path.display().to_string());
            }

            let mut proof_written: Option<String> = None;
            if let Some(path) = emit_proof.as_ref() {
                // Only write if we have the full proof text (avoid writing a likely-invalid truncated sexp).
                let pf_ok = proof.as_ref().ok().and_then(|x| x.as_ref()).and_then(|pf| {
                    let preview = pf.get("preview")?.as_str()?;
                    let chars = pf.get("chars")?.as_u64()? as usize;
                    let preview_chars = preview.chars().count();
                    if chars <= proof_max_chars && preview_chars == chars {
                        Some(preview.to_string())
                    } else {
                        None
                    }
                });
                if let Some(txt) = pf_ok {
                    if let Some(parent) = path.parent() {
                        std::fs::create_dir_all(parent).map_err(|e| {
                            format!("failed to create dir {}: {e}", parent.display())
                        })?;
                    }
                    std::fs::write(path, txt.as_bytes())
                        .map_err(|e| format!("failed to write {}: {e}", path.display()))?;
                    proof_written = Some(path.display().to_string());
                }
            }

            let goals_n = pp_dump
                .get("goals")
                .and_then(|g| g.as_array())
                .map(|g| g.len())
                .unwrap_or(0);
            let result_kind =
                if solver_probe.get("available").and_then(|v| v.as_bool()) != Some(true) {
                    "solver_unavailable"
                } else if smt2.is_some() {
                    "ok"
                } else {
                    "no_entailment"
                };
            let note = match result_kind {
                "solver_unavailable" => "no SMT solver detected by probe",
                "no_entailment" => {
                    "pp_dump parsed, but the goal does not look like a supported SMT-LIA entailment"
                }
                _ => "",
            };

            let capsule = if let Some(dir) = bundle_dir.as_ref() {
                // Always write pp_dump.json (even if no entailment).
                let pp_path = dir.join("pp_dump.json");
                write_json(&pp_path, &pp_dump)?;

                // Write a small manifest with stable identifiers (hashes) and environment notes.
                let pp_bytes = serde_json::to_vec(&pp_dump)
                    .map_err(|e| format!("failed to encode pp_dump for hashing: {e}"))?;
                let smt2_sha256 = smt2.as_ref().map(|s| sha256_hex(s.as_bytes()));
                let proof_sha256 = proof
                    .as_ref()
                    .ok()
                    .and_then(|x| x.as_ref())
                    .and_then(|pf| pf.get("preview").and_then(|v| v.as_str()))
                    .map(|s| sha256_hex(s.as_bytes()));
                let manifest = json!({
                    "kind": "proofpatch_smt_capsule",
                    "tool": {
                        "proofpatch_cli_version": env!("CARGO_PKG_VERSION"),
                    },
                    "input": {
                        "input_json": input_label,
                        "pp_dump_path": pp_path.display().to_string(),
                        "pp_dump_sha256": sha256_hex(&pp_bytes),
                        "pp_dump_goals": goals_n,
                    },
                    "params": {
                        "timeout_ms": timeout_ms,
                        "seed": seed,
                        "depth": depth,
                        "proof_max_chars": proof_max_chars,
                    },
                    "solver": solver_probe,
                    "result": {
                        "result_kind": result_kind,
                        "note": if note.is_empty() { serde_json::Value::Null } else { json!(note) },
                        "smt2_sha256": smt2_sha256,
                        "proof_sha256": proof_sha256,
                    }
                });
                let manifest_path = dir.join("manifest.json");
                write_json(&manifest_path, &manifest)?;
                json!({
                    "bundle_dir": dir.display().to_string(),
                    "pp_dump": pp_path.display().to_string(),
                    "manifest": manifest_path.display().to_string(),
                })
            } else {
                serde_json::Value::Null
            };

            let out = json!({
                "ok": true,
                "kind": "smt_repro",
                "result_kind": result_kind,
                "input_json": input_label,
                "solver": solver_probe,
                "params": {
                    "timeout_ms": timeout_ms,
                    "seed": seed,
                    "depth": depth,
                    "proof_max_chars": proof_max_chars,
                },
                "pp_dump": {
                    "goals": goals_n,
                },
                "note": if note.is_empty() { serde_json::Value::Null } else { json!(note) },
                "capsule": capsule,
                "artifacts": {
                    "smt2_requested": emit_smt2.as_ref().map(|p| p.display().to_string()),
                    "smt2_written": smt2_written,
                    "proof_requested": emit_proof.as_ref().map(|p| p.display().to_string()),
                    "proof_written": proof_written,
                },
                "smt2": smt2.unwrap_or_else(|| "".to_string()),
                "proof": match proof {
                    Ok(pf) => pf.unwrap_or(serde_json::Value::Null),
                    Err(e) => json!({"error": truncate_str(&e, 400)}),
                }
            });

            if let Some(p) = output_json {
                write_json(&p, &out)?;
                println!(
                    "{}",
                    json!({
                        "ok": true,
                        "written": p.display().to_string(),
                        "kind": "smt_repro",
                        "result_kind": out["result_kind"],
                    })
                    .to_string()
                );
            } else {
                println!("{}", out.to_string());
            }
            Ok(())
        }

        "suggest" => {
            let repo_root = arg_value(rest, "--repo")
                .ok_or_else(|| "missing --repo".to_string())
                .map(PathBuf::from)?;
            let file = arg_value(rest, "--file").ok_or_else(|| "missing --file".to_string())?;
            let lemma = arg_value(rest, "--lemma").ok_or_else(|| "missing --lemma".to_string())?;
            let timeout_s = arg_u64(rest, "--timeout-s").unwrap_or(120);
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);

            let repo_root =
                plc::find_lean_repo_root(&repo_root).map_err(|e| format!("repo_root: {e}"))?;
            plc::load_dotenv_smart(&repo_root);

            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| format!("failed to build tokio runtime: {e}"))?;

            let payload = plc::build_proof_prompt(&repo_root, &file, &lemma)?;
            let res = rt
                .block_on(plc::llm::chat_completion(
                    &payload.system,
                    &payload.user,
                    StdDuration::from_secs(timeout_s),
                ))
                .map_err(|e| format!("llm chat_completion failed: {e}"))?;

            let out = json!({
                "provider": res.provider,
                "model": res.model,
                "lemma": lemma,
                "file": payload.file,
                "prompt": {
                    "combined_chars": payload.prompt_combined_chars,
                    "combined_sha256": payload.prompt_combined_sha256,
                },
                "suggestion": res.content,
                "raw": res.raw
            });

            if let Some(p) = output_json {
                write_json(&p, &out)?;
                println!(
                    "{}",
                    json!({
                        "ok": true,
                        "written": p.display().to_string(),
                        "kind": "suggest",
                        "result_kind": serde_json::Value::Null,
                    })
                    .to_string()
                );
            } else {
                println!("{}", out.to_string());
            }
            Ok(())
        }

        "loop" => {
            let repo_root = arg_value(rest, "--repo")
                .ok_or_else(|| "missing --repo".to_string())
                .map(PathBuf::from)?;
            let file = arg_value(rest, "--file").ok_or_else(|| "missing --file".to_string())?;
            let lemma = arg_value(rest, "--lemma").ok_or_else(|| "missing --lemma".to_string())?;
            let max_iters = arg_u64(rest, "--max-iters").unwrap_or(3);
            let timeout_s = arg_u64(rest, "--timeout-s").unwrap_or(120);
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);

            if max_iters == 0 {
                return Err("max-iters must be >= 1".to_string());
            }

            let repo_root =
                plc::find_lean_repo_root(&repo_root).map_err(|e| format!("repo_root: {e}"))?;
            plc::load_dotenv_smart(&repo_root);

            let p = repo_root.join(&file);
            if !p.exists() {
                return Err(format!("File not found: {}", p.display()));
            }
            let mut cur_text =
                std::fs::read_to_string(&p).map_err(|e| format!("read {}: {e}", p.display()))?;

            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| format!("failed to build tokio runtime: {e}"))?;

            let mut attempts: Vec<serde_json::Value> = Vec::new();
            for iter_idx in 0..max_iters {
                let excerpt = plc::extract_decl_block(&cur_text, &lemma)?;
                let system = plc::proof_system_prompt();
                let user = plc::proof_user_prompt(&excerpt);

                let res = rt
                    .block_on(plc::llm::chat_completion(
                        &system,
                        &user,
                        StdDuration::from_secs(timeout_s),
                    ))
                    .map_err(|e| format!("llm chat_completion failed: {e}"))?;

                let suggestion = json!({
                    "provider": res.provider,
                    "model": res.model,
                    "lemma": lemma,
                    "file": p.display().to_string(),
                    "suggestion": res.content,
                    "raw": res.raw
                });

                let replacement = suggestion
                    .get("suggestion")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        "LLM suggestion did not contain `suggestion` field".to_string()
                    })?;

                let patched = plc::patch_first_sorry_in_decl(&cur_text, &lemma, replacement)?;
                cur_text = patched.text.clone();

                let still_has_sorry = plc::decl_block_contains_sorry(&cur_text, &lemma)?;
                let verify = rt
                    .block_on(plc::verify_lean_text(
                        &repo_root,
                        &cur_text,
                        StdDuration::from_secs(timeout_s),
                    ))
                    .map_err(|e| format!("verify failed: {e}"))?;

                attempts.push(json!({
                    "iter": iter_idx + 1,
                    "suggestion": suggestion,
                    "patch": {
                        "line": patched.line,
                        "before": patched.before,
                        "after": patched.after,
                        "indent": patched.indent,
                    },
                    "lemma_still_contains_sorry": still_has_sorry,
                    "verify": verify,
                }));

                if verify.ok && !still_has_sorry {
                    break;
                }
            }

            let final_still_has_sorry =
                plc::decl_block_contains_sorry(&cur_text, &lemma).unwrap_or(true);
            let out = json!({
                "file": p.display().to_string(),
                "lemma": lemma,
                "max_iters": max_iters,
                "attempts": attempts,
                "final_lemma_contains_sorry": final_still_has_sorry,
            });

            if let Some(p) = output_json {
                write_json(&p, &out)?;
                println!(
                    "{}",
                    json!({
                        "ok": true,
                        "written": p.display().to_string(),
                        "kind": "loop",
                        "result_kind": serde_json::Value::Null,
                    })
                    .to_string()
                );
            } else {
                println!("{}", out.to_string());
            }
            Ok(())
        }

        "review-prompt" => {
            let repo_root = arg_value(rest, "--repo")
                .ok_or_else(|| "missing --repo".to_string())
                .map(PathBuf::from)?;
            // Default: `auto` (same semantics as `review-diff`):
            // - if staged has changes, return staged prompt (commit-focused)
            // - otherwise return worktree prompt (more useful during iteration)
            let scope = arg_value(rest, "--scope").unwrap_or_else(|| "auto".to_string());
            let max_total_bytes = arg_u64(rest, "--max-total-bytes").unwrap_or(180_000) as usize;
            let per_file_bytes = arg_u64(rest, "--per-file-bytes").unwrap_or(24_000) as usize;
            let transcript_bytes = arg_u64(rest, "--transcript-bytes").unwrap_or(24_000) as usize;
            let cache_version =
                arg_value(rest, "--cache-version").unwrap_or_else(|| "2026-01-19-v1".to_string());
            let cache_model =
                arg_value(rest, "--cache-model").unwrap_or_else(|| "unknown".to_string());
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);

            let scope_req = scope.trim().to_lowercase();
            if !matches!(scope_req.as_str(), "auto" | "staged" | "worktree") {
                return Err("scope must be auto|staged|worktree".to_string());
            }

            // Best-effort: keep progress off by default unless explicitly enabled.
            // (Review prompt building can run in pre-commit contexts.)
            let prompt = if scope_req == "auto" {
                let p_staged = plc::review::build_review_prompt(
                    &repo_root,
                    plc::review::ReviewScope::Staged,
                    max_total_bytes,
                    per_file_bytes,
                    transcript_bytes,
                    &cache_model,
                    &cache_version,
                )?;
                if p_staged.diff.trim().is_empty() && p_staged.selected_files.is_empty() {
                    plc::review::build_review_prompt(
                        &repo_root,
                        plc::review::ReviewScope::Worktree,
                        max_total_bytes,
                        per_file_bytes,
                        transcript_bytes,
                        &cache_model,
                        &cache_version,
                    )?
                } else {
                    p_staged
                }
            } else {
                let scope_enum = match scope_req.as_str() {
                    "staged" => plc::review::ReviewScope::Staged,
                    "worktree" => plc::review::ReviewScope::Worktree,
                    _ => unreachable!("validated above"),
                };
                plc::review::build_review_prompt(
                    &repo_root,
                    scope_enum,
                    max_total_bytes,
                    per_file_bytes,
                    transcript_bytes,
                    &cache_model,
                    &cache_version,
                )?
            };
            let out = serde_json::to_value(prompt).map_err(|e| format!("json encode: {e}"))?;

            if let Some(p) = output_json {
                write_json(&p, &out)?;
                println!(
                    "{}",
                    json!({
                        "ok": true,
                        "written": p.display().to_string(),
                        "kind": "review_prompt",
                        "result_kind": serde_json::Value::Null,
                    })
                    .to_string()
                );
            } else {
                println!("{}", out.to_string());
            }
            Ok(())
        }

        "review-diff" => {
            let repo_root = arg_value(rest, "--repo")
                .ok_or_else(|| "missing --repo".to_string())
                .map(PathBuf::from)?;
            // Default: `auto`:
            // - if staged has changes, review staged (commit-focused)
            // - otherwise review worktree (more useful during iteration)
            let scope = arg_value(rest, "--scope").unwrap_or_else(|| "auto".to_string());
            let prompt_only = arg_flag(rest, "--prompt-only");
            let require_key = arg_flag(rest, "--require-key");
            let timeout_s = arg_u64(rest, "--timeout-s").unwrap_or(90);
            let no_verify = arg_flag(rest, "--no-verify");
            let verify_timeout_s = arg_u64(rest, "--verify-timeout-s").unwrap_or(60);
            let verify_max_files = arg_u64(rest, "--verify-max-files").unwrap_or(6) as usize;
            let max_total_bytes = arg_u64(rest, "--max-total-bytes").unwrap_or(180_000) as usize;
            let per_file_bytes = arg_u64(rest, "--per-file-bytes").unwrap_or(24_000) as usize;
            let transcript_bytes = arg_u64(rest, "--transcript-bytes").unwrap_or(24_000) as usize;
            let cache_version =
                arg_value(rest, "--cache-version").unwrap_or_else(|| "2026-01-19-v1".to_string());
            let cache_model =
                arg_value(rest, "--cache-model").unwrap_or_else(|| "unknown".to_string());
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);

            let scope_req = scope.trim().to_lowercase();
            if !matches!(scope_req.as_str(), "auto" | "staged" | "worktree") {
                return Err("scope must be auto|staged|worktree".to_string());
            }

            // Determine git root for env loading (LLM keys often live in <repo>/.env).
            let git_root = plc::review::git_repo_root(&repo_root)?;
            plc::load_dotenv_smart(&git_root);

            let (prompt, resolved_scope) = if scope_req == "auto" {
                let p_staged = plc::review::build_review_prompt(
                    &git_root,
                    plc::review::ReviewScope::Staged,
                    max_total_bytes,
                    per_file_bytes,
                    transcript_bytes,
                    &cache_model,
                    &cache_version,
                )?;
                if p_staged.diff.trim().is_empty() && p_staged.selected_files.is_empty() {
                    let p_wt = plc::review::build_review_prompt(
                        &git_root,
                        plc::review::ReviewScope::Worktree,
                        max_total_bytes,
                        per_file_bytes,
                        transcript_bytes,
                        &cache_model,
                        &cache_version,
                    )?;
                    (p_wt, "worktree".to_string())
                } else {
                    (p_staged, "staged".to_string())
                }
            } else {
                let scope_enum = match scope_req.as_str() {
                    "staged" => plc::review::ReviewScope::Staged,
                    "worktree" => plc::review::ReviewScope::Worktree,
                    _ => unreachable!("validated above"),
                };
                let p = plc::review::build_review_prompt(
                    &git_root,
                    scope_enum,
                    max_total_bytes,
                    per_file_bytes,
                    transcript_bytes,
                    &cache_model,
                    &cache_version,
                )?;
                (p, scope_req.clone())
            };

            // Redact before emitting or sending.
            let diff = plc::review::redact_secrets(&prompt.diff);
            let corpus = plc::review::redact_secrets(&prompt.corpus);
            let transcript_tail = plc::review::redact_secrets(&prompt.transcript_tail);

            // Structured review: keep it machine-readable so scripts can print a short summary.
            // (Humans can still read the JSON; no markdown fences.)
            let system = [
                "You are a skeptical code reviewer for a Lean/mathlib-focused repository.",
                "You are given: (a) a git diff, (b) bounded file excerpts, and (c) optional Lean verify summaries.",
                "Treat verify summaries as ground truth for whether a file parses / elaborates.",
                "Priorities:",
                "- correctness and proof soundness",
                "- API stability / portability (Linux case sensitivity, CI)",
                "- minimal diffs (prefer small fixes)",
                "Avoid praise. If unsure, say what is missing.",
                "",
                "Important constraints:",
                "- Some file excerpts may be marked TRUNCATED (prefix+suffix). Do not infer a parse error from missing text.",
                "- Do not claim \"file is truncated\" unless the input explicitly says so.",
                "- If verify.ok=true for a file, do not claim it fails to parse/elaborate.",
                "",
                "Return STRICT JSON (no markdown) matching this shape:",
                "{",
                "  \"overall\": {",
                "    \"score\": 0-100,",
                "    \"verdict\": \"approve|request_changes|needs_context\",",
                "    \"summary\": \"1-2 sentences\"",
                "  },",
                "  \"axes\": [",
                "    {\"name\":\"correctness\",\"score\":0-100,\"summary\":\"...\"},",
                "    {\"name\":\"proof_soundness\",\"score\":0-100,\"summary\":\"...\"},",
                "    {\"name\":\"maintainability\",\"score\":0-100,\"summary\":\"...\"},",
                "    {\"name\":\"risk\",\"score\":0-100,\"summary\":\"higher = riskier\"}",
                "  ],",
                "  \"top_issues\": [",
                "    {\"severity\":\"high|medium|low\",\"title\":\"...\",\"detail\":\"...\",\"files\":[\"...\"]}",
                "  ],",
                "  \"quick_wins\": [\"...\"],",
                "  \"questions\": [\"...\"],",
                "  \"confidence\": 0.0-1.0",
                "}",
            ]
            .join("\n");

            // Optional: run Lean verification on changed `.lean` files.
            let verify = if prompt_only || no_verify {
                serde_json::Value::Null
            } else if let Ok(lean_root) = plc::find_lean_repo_root(&git_root) {
                let rt = tokio::runtime::Runtime::new()
                    .map_err(|e| format!("failed to build tokio runtime: {e}"))?;
                let mut rows = Vec::new();
                for f in prompt
                    .selected_files
                    .iter()
                    .filter(|p| p.ends_with(".lean"))
                    .take(verify_max_files)
                {
                    let raw = rt
                        .block_on(plc::verify_lean_file(
                            &lean_root,
                            f,
                            StdDuration::from_secs(verify_timeout_s),
                        ))
                        .map_err(|e| format!("verify failed for {f}: {e}"))?;
                    let raw_v =
                        serde_json::to_value(raw).map_err(|e| format!("serialize verify: {e}"))?;
                    let ok = raw_v.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
                    let stdout = raw_v.get("stdout").and_then(|v| v.as_str()).unwrap_or("");
                    let stderr = raw_v.get("stderr").and_then(|v| v.as_str()).unwrap_or("");
                    let errors =
                        stdout.matches(": error:").count() + stderr.matches(": error:").count();
                    let warnings =
                        stdout.matches(": warning:").count() + stderr.matches(": warning:").count();
                    let error_samples: Vec<String> = stdout
                        .lines()
                        .chain(stderr.lines())
                        .filter(|l| l.contains(": error:"))
                        .take(6)
                        .map(|s| s.to_string())
                        .collect();
                    let warning_samples: Vec<String> = stdout
                        .lines()
                        .chain(stderr.lines())
                        .filter(|l| l.contains(": warning:"))
                        .take(6)
                        .map(|s| s.to_string())
                        .collect();
                    rows.push(json!({
                        "file": f,
                        "ok": ok,
                        "errors": errors,
                        "warnings": warnings,
                        "error_samples": error_samples,
                        "warning_samples": warning_samples,
                    }));
                }
                json!({
                    "lean_repo_root": lean_root.display().to_string(),
                    "timeout_s": verify_timeout_s,
                    "max_files": verify_max_files,
                    "files": rows,
                })
            } else {
                serde_json::Value::Null
            };

            let payload = json!({
                "repo_root": prompt.repo_root,
                "scope": prompt.scope,
                "scope_requested": scope_req,
                "scope_resolved": resolved_scope,
                "selected_files": prompt.selected_files,
                "blob_meta": prompt.blob_meta,
                "diff": diff,
                "corpus": corpus,
                "transcript_tail": transcript_tail,
                "verify": verify,
                "cache_key": prompt.cache_key,
                "max_total_bytes": prompt.max_total_bytes,
            });

            let user = serde_json::to_string(&payload).map_err(|e| format!("json encode: {e}"))?;

            if prompt_only {
                let out = json!({
                    "repo_root": payload.get("repo_root").cloned().unwrap_or(serde_json::Value::Null),
                    "scope": payload.get("scope").cloned().unwrap_or(serde_json::Value::Null),
                    "system": system,
                    "payload": payload,
                    "user": user,
                });
                if let Some(p) = output_json {
                    write_json(&p, &out)?;
                    println!(
                        "{}",
                        json!({
                            "ok": true,
                            "written": p.display().to_string(),
                            "kind": "review_diff",
                            "result_kind": serde_json::Value::Null,
                        })
                        .to_string()
                    );
                } else {
                    println!("{}", out.to_string());
                }
                return Ok(());
            }

            // Avoid calling an LLM when there is nothing to review.
            let diff_empty = payload
                .get("diff")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .is_empty();
            let files_empty = payload
                .get("selected_files")
                .and_then(|v| v.as_array())
                .map(|a| a.is_empty())
                .unwrap_or(true);
            if diff_empty && files_empty {
                let out = json!({
                    "skipped": true,
                    "reason": format!("no changes found for scope={}", payload.get("scope").and_then(|v| v.as_str()).unwrap_or("unknown")),
                    "repo_root": payload.get("repo_root").cloned().unwrap_or(serde_json::Value::Null),
                    "scope": payload.get("scope").cloned().unwrap_or(serde_json::Value::Null),
                    "scope_requested": payload.get("scope_requested").cloned().unwrap_or(serde_json::Value::Null),
                    "scope_resolved": payload.get("scope_resolved").cloned().unwrap_or(serde_json::Value::Null),
                    "cache_key": payload.get("cache_key").cloned().unwrap_or(serde_json::Value::Null),
                });
                if let Some(p) = output_json {
                    write_json(&p, &out)?;
                    println!(
                        "{}",
                        json!({
                            "ok": true,
                            "written": p.display().to_string(),
                            "kind": "review_diff",
                            "result_kind": serde_json::Value::Null,
                        })
                        .to_string()
                    );
                } else {
                    println!("{}", out.to_string());
                }
                return Ok(());
            }

            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| format!("failed to build tokio runtime: {e}"))?;

            let res = rt.block_on(plc::llm::chat_completion(
                &system,
                &user,
                StdDuration::from_secs(timeout_s),
            ));

            let out = match res {
                Ok(r) => {
                    // Best-effort: extract the first JSON value from the model output.
                    let review_struct = extract_json_from_text(&r.content);
                    let mut review_struct = review_struct;
                    let mut postprocess_notes: Vec<String> = Vec::new();

                    // If we ran Lean verification and it says a file is OK, suppress any
                    // “won't parse/compile because truncated” claims for that file.
                    if let Some(v) = payload.get("verify") {
                        if let Some(files) = v.get("files").and_then(|x| x.as_array()) {
                            let mut ok_files: std::collections::BTreeSet<String> =
                                std::collections::BTreeSet::new();
                            for row in files {
                                let ok = row.get("ok").and_then(|x| x.as_bool()).unwrap_or(false);
                                let errors =
                                    row.get("errors").and_then(|x| x.as_u64()).unwrap_or(0);
                                if ok && errors == 0 {
                                    if let Some(f) = row.get("file").and_then(|x| x.as_str()) {
                                        ok_files.insert(f.to_string());
                                    }
                                }
                            }

                            if !ok_files.is_empty() {
                                if let Some(rs) = review_struct.as_mut() {
                                    // top_issues: drop “truncation/parse/compile” claims for ok files.
                                    if let Some(arr) =
                                        rs.get_mut("top_issues").and_then(|x| x.as_array_mut())
                                    {
                                        let before = arr.len();
                                        arr.retain(|issue| {
                                            let title = issue
                                                .get("title")
                                                .and_then(|x| x.as_str())
                                                .unwrap_or("");
                                            let detail = issue
                                                .get("detail")
                                                .and_then(|x| x.as_str())
                                                .unwrap_or("");
                                            let t = format!("{title}\n{detail}").to_lowercase();
                                            let mentions_build_break = t.contains("truncat")
                                                || t.contains("parse")
                                                || t.contains("elaborat")
                                                || t.contains("compile")
                                                || t.contains("will not build")
                                                || t.contains("won't build")
                                                || t.contains("will break the build");
                                            if !mentions_build_break {
                                                return true;
                                            }
                                            let hits_ok = issue
                                                .get("files")
                                                .and_then(|x| x.as_array())
                                                .map(|files| {
                                                    files.iter().any(|f| {
                                                        f.as_str()
                                                            .map(|s| ok_files.contains(s))
                                                            .unwrap_or(false)
                                                    })
                                                })
                                                .unwrap_or(false);
                                            !hits_ok
                                        });
                                        let after = arr.len();
                                        if after != before {
                                            postprocess_notes.push(format!(
                                                "removed {} top_issues that claimed parse/compile failure despite verify.ok=true",
                                                before - after
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    json!({
                    "repo_root": payload.get("repo_root").cloned().unwrap_or(serde_json::Value::Null),
                    "scope": payload.get("scope").cloned().unwrap_or(serde_json::Value::Null),
                    "scope_requested": payload.get("scope_requested").cloned().unwrap_or(serde_json::Value::Null),
                    "scope_resolved": payload.get("scope_resolved").cloned().unwrap_or(serde_json::Value::Null),
                    "selected_files": payload.get("selected_files").cloned().unwrap_or(serde_json::Value::Null),
                    "blob_meta": payload.get("blob_meta").cloned().unwrap_or(serde_json::Value::Null),
                    "verify": payload.get("verify").cloned().unwrap_or(serde_json::Value::Null),
                    "provider": r.provider,
                    "model": r.model,
                    "model_source": r.model_source,
                    "model_env": r.model_env,
                    "review_text": r.content,
                    "review_struct": review_struct,
                    "postprocess_notes": postprocess_notes,
                    "cache_key": payload.get("cache_key").cloned().unwrap_or(serde_json::Value::Null),
                    })
                }
                Err(e) => {
                    if require_key {
                        return Err(format!("review-diff failed: {e}"));
                    }
                    // Keep "skip" behavior non-fatal, but structured for agents.
                    json!({
                        "skipped": true,
                        "reason": e,
                        "repo_root": payload.get("repo_root").cloned().unwrap_or(serde_json::Value::Null),
                        "scope": payload.get("scope").cloned().unwrap_or(serde_json::Value::Null),
                        "scope_requested": payload.get("scope_requested").cloned().unwrap_or(serde_json::Value::Null),
                        "scope_resolved": payload.get("scope_resolved").cloned().unwrap_or(serde_json::Value::Null),
                        "cache_key": payload.get("cache_key").cloned().unwrap_or(serde_json::Value::Null),
                    })
                }
            };

            if let Some(p) = output_json {
                write_json(&p, &out)?;
                println!(
                    "{}",
                    json!({
                        "ok": true,
                        "written": p.display().to_string(),
                        "kind": "review_diff",
                        "result_kind": serde_json::Value::Null,
                    })
                    .to_string()
                );
            } else {
                println!("{}", out.to_string());
            }
            Ok(())
        }

        "llm-chat" => {
            let repo_root = arg_value(rest, "--repo").map(PathBuf::from);
            let system = arg_value(rest, "--system");
            let system_file = arg_value(rest, "--system-file").map(PathBuf::from);
            let user = arg_value(rest, "--user");
            let user_file = arg_value(rest, "--user-file").map(PathBuf::from);
            let require_key = arg_flag(rest, "--require-key");
            let timeout_s = arg_u64(rest, "--timeout-s").unwrap_or(90);
            let _max_tool_iters = arg_u64(rest, "--max-tool-iters").unwrap_or(4) as usize;
            let tools = arg_value(rest, "--tools").unwrap_or_default();
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);

            let system_txt = if let Some(p) = system_file {
                fs::read_to_string(&p).map_err(|e| format!("read {}: {}", p.display(), e))?
            } else {
                system.unwrap_or_default()
            };
            let user_txt = if let Some(p) = user_file {
                fs::read_to_string(&p).map_err(|e| format!("read {}: {}", p.display(), e))?
            } else {
                user.unwrap_or_default()
            };
            if system_txt.trim().is_empty() && user_txt.trim().is_empty() {
                return Err(
                    "llm-chat requires --system/--system-file and/or --user/--user-file"
                        .to_string(),
                );
            }
            let system_txt = if system_txt.trim().is_empty() {
                // Default: treat user input as structured JSON from proofpatch and respond tersely.
                [
                    "You are a Lean 4/mathlib proof assistant and reviewer.",
                    "You will be given either plain text or JSON from a tool (often proofpatch rubberduck-prompt).",
                    "Extract the current goal(s), key hypotheses, and the smallest next lemma steps.",
                    "Prefer reusing existing lemmas in the repository or mathlib over inventing new ones.",
                    "If tools are available, use them to fetch missing context instead of guessing.",
                    "Return STRICT JSON (no markdown) with keys:",
                    r#"{"goal":"...","plan":["..."],"lean_steps":["..."],"math_notes":["..."],"questions":["..."]}"#,
                ]
                .join("\n")
            } else {
                system_txt
            };

            // Load env from --repo git root if provided, else from current dir's git root.
            if let Some(rr) = repo_root.as_ref() {
                if let Ok(git_root) = plc::review::git_repo_root(rr) {
                    plc::load_dotenv_smart(&git_root);
                } else {
                    plc::load_dotenv_smart(rr);
                }
            } else if let Ok(cwd) = std::env::current_dir() {
                if let Ok(git_root) = plc::review::git_repo_root(&cwd) {
                    plc::load_dotenv_smart(&git_root);
                }
            }

            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| format!("failed to build tokio runtime: {e}"))?;

            let tools = tools.trim();
            if !tools.is_empty() {
                return Err(
                    "llm-chat --tools is not supported in the standalone proofpatch build (tool-calling backend not included)"
                        .to_string(),
                );
            }
            let out = {
                // Simple one-shot chat completion (no tools).
                let res0 = rt.block_on(plc::llm::chat_completion(
                    &system_txt,
                    &user_txt,
                    StdDuration::from_secs(timeout_s),
                ));
                match res0 {
                    Ok(r) => json!({
                        "ok": true,
                        "provider": r.provider,
                        "model": r.model,
                        "model_source": r.model_source,
                        "model_env": r.model_env,
                        "content": r.content,
                        "content_struct": extract_json_from_text(&r.content),
                        "raw": r.raw,
                    }),
                    Err(e) => {
                        if require_key {
                            return Err(format!("llm-chat failed: {e}"));
                        }
                        json!({"skipped": true, "reason": e})
                    }
                }
            };

            if let Some(p) = output_json {
                write_json(&p, &out)?;
                println!(
                    "{}",
                    json!({
                        "ok": true,
                        "written": p.display().to_string(),
                        "kind": "llm_chat",
                        "result_kind": serde_json::Value::Null,
                    })
                    .to_string()
                );
            } else {
                println!("{}", out.to_string());
            }
            Ok(())
        }

        "lint-style" => {
            let repo_root = arg_value(rest, "--repo")
                .ok_or_else(|| "missing --repo".to_string())
                .map(PathBuf::from)?;
            let github = arg_flag(rest, "--github");
            let modules = arg_values(rest, "--module");
            if modules.is_empty() {
                return Err("lint-style requires at least one --module <Root>".to_string());
            }

            let repo_root =
                plc::find_lean_repo_root(&repo_root).map_err(|e| format!("repo_root: {e}"))?;
            let lake = plc::resolve_lake();

            let mut cmd = std::process::Command::new(lake);
            cmd.arg("exe").arg("lint-style");
            if github {
                cmd.arg("--github");
            }
            for m in modules {
                cmd.arg(m);
            }
            let status = cmd
                .current_dir(&repo_root)
                .status()
                .map_err(|e| format!("failed to run lake lint-style: {e}"))?;
            if status.success() {
                Ok(())
            } else {
                Err(format!("lint-style failed with status: {status}"))
            }
        }

        "report" => {
            let repo_root = arg_value(rest, "--repo")
                .ok_or_else(|| "missing --repo".to_string())
                .map(PathBuf::from)?;

            // Files are positional after `--files`.
            let files_idx = rest
                .iter()
                .position(|a| a == "--files")
                .ok_or_else(|| "missing --files".to_string())?;
            let files: Vec<String> = rest[(files_idx + 1)..]
                .iter()
                .take_while(|s| !s.starts_with("--"))
                .cloned()
                .collect();
            if files.is_empty() {
                return Err("empty --files list".to_string());
            }

            let timeout_s = arg_u64(rest, "--timeout-s").unwrap_or(180);
            let max_sorries = arg_u64(rest, "--max-sorries").unwrap_or(5) as usize;
            let context_lines = arg_u64(rest, "--context-lines").unwrap_or(1) as usize;
            let include_raw_verify = arg_flag(rest, "--include-raw-verify");
            let output_html = arg_value(rest, "--output-html").map(PathBuf::from);

            let repo_root =
                plc::find_lean_repo_root(&repo_root).map_err(|e| format!("repo_root: {e}"))?;
            plc::load_dotenv_smart(&repo_root);

            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| format!("failed to build tokio runtime: {e}"))?;

            let mut table = Vec::new();
            let mut items = Vec::new();
            let mut next_actions = Vec::new();

            for file in &files {
                let raw = rt
                    .block_on(plc::verify_lean_file(
                        &repo_root,
                        file,
                        StdDuration::from_secs(timeout_s),
                    ))
                    .map_err(|e| format!("verify failed for {file}: {e}"))?;
                let raw_v =
                    serde_json::to_value(raw).map_err(|e| format!("serialize verify: {e}"))?;

                let ok = raw_v.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
                let stdout = raw_v.get("stdout").and_then(|v| v.as_str()).unwrap_or("");
                let stderr = raw_v.get("stderr").and_then(|v| v.as_str()).unwrap_or("");
                let errors =
                    stdout.matches(": error:").count() + stderr.matches(": error:").count();
                let warnings =
                    stdout.matches(": warning:").count() + stderr.matches(": warning:").count();
                let error_samples: Vec<String> = stdout
                    .lines()
                    .chain(stderr.lines())
                    .filter(|l| l.contains(": error:"))
                    .take(8)
                    .map(|s| s.to_string())
                    .collect();
                let warning_samples: Vec<String> = stdout
                    .lines()
                    .chain(stderr.lines())
                    .filter(|l| l.contains(": warning:"))
                    .take(8)
                    .map(|s| s.to_string())
                    .collect();

                let locs =
                    plc::locate_sorries_in_file(&repo_root, file, max_sorries, context_lines)?;
                let conservative_sorries =
                    plc::count_sorry_tokens_conservative_in_file(&repo_root, file).unwrap_or(0);

                // Flatten into agent-ready next actions.
                for e in &error_samples {
                    let lean_query = format!("Lean 4 {}", e);
                    next_actions.push(json!({
                        "kind": "fix_error",
                        "file": file,
                        "message": e,
                        "research": {
                            "plan": {
                                "goal": "Find the underlying cause and a minimal fix.",
                                "calls": [
                                {
                                    "server": "user-arxiv-semantic-search-mcp",
                                    "toolName": "search_papers",
                                    "arguments": { "query": lean_query }
                                },
                                {
                                    "server": "user-tavily-remote-mcp",
                                    "toolName": "tavily_search",
                                    "arguments": {
                                        "query": lean_query,
                                        "search_depth": "advanced",
                                        "max_results": 5
                                    }
                                },
                                {
                                    "server": "user-perplexity",
                                    "toolName": "search",
                                    "arguments": {
                                        "query": lean_query
                                    }
                                }
                                ],
                                "extract": {
                                    "schema": {
                                        "type": "object",
                                        "additionalProperties": false,
                                        "properties": {
                                            "root_cause": { "type": "string" },
                                            "minimal_fix": { "type": "string" },
                                            "keywords": { "type": "array", "items": { "type": "string" } },
                                            "sources": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "additionalProperties": false,
                                                    "properties": {
                                                        "title": { "type": "string" },
                                                        "url": { "type": "string" }
                                                    },
                                                    "required": ["url"]
                                                }
                                            }
                                        },
                                        "required": ["root_cause", "minimal_fix"]
                                    }
                                }
                            }
                        }
                    }));
                }
                for loc in &locs {
                    // Domain-ish queries keyed off the enclosing declaration name when we have it.
                    let decl = loc
                        .decl_name
                        .clone()
                        .unwrap_or_else(|| "unknown_decl".to_string());
                    let q = if decl.contains("nathanson") || decl.contains("polygonal") {
                        "Fermat polygonal number theorem Nathanson proof b^2 < 4a 3a < b^2 + 2b + 4 Cauchy lemma"
                            .to_string()
                    } else if decl.contains("cauchy_lemma") {
                        "Cauchy lemma b^2 < 4a 0 < b^2 + 2b - 3a + 4 a = sum of four squares b = sum of variables"
                            .to_string()
                    } else if decl.contains("sum_three_squares") || decl.contains("Legendre") {
                        "sum of three squares theorem residue classes mod 8 remaining cases 2 5 6"
                            .to_string()
                    } else {
                        format!("Lean proof {}", decl)
                    };
                    let web_q = format!("{q} mathlib Lean");
                    let extract_schema = json!({
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "math_statement": { "type": "string" },
                            "variables": {
                                "type": "object",
                                "additionalProperties": { "type": "string" }
                            },
                            "constraints": { "type": "array", "items": { "type": "string" } },
                            "candidate_mathlib_lemmas": { "type": "array", "items": { "type": "string" } },
                            "sources": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": false,
                                    "properties": {
                                        "title": { "type": "string" },
                                        "url": { "type": "string" }
                                    },
                                    "required": ["url"]
                                }
                            }
                        },
                        "required": ["math_statement"]
                    });

                    next_actions.push(json!({
                        "kind": "fix_sorry",
                        "file": file,
                        "token": loc.token,
                        "line": loc.line,
                        "decl_kind": loc.decl_kind,
                        "decl_name": loc.decl_name,
                        "decl_line": loc.decl_line,
                        "excerpt": loc.excerpt,
                        "research": {
                            "plan": {
                                "goal": "Find a correct mathematical step and map it onto Lean/mathlib lemmas.",
                                "calls": [
                                {
                                    "server": "user-arxiv-semantic-search-mcp",
                                    "toolName": "search_papers",
                                    "arguments": { "query": q }
                                },
                                {
                                    "server": "user-firecrawl-mcp",
                                    "toolName": "firecrawl_search",
                                    "arguments": {
                                        "query": web_q,
                                        "limit": 5,
                                        "sources": [{"type": "web"}],
                                        "scrapeOptions": {
                                            "formats": [
                                                {
                                                    "type": "json",
                                                    "prompt": "Extract: math_statement, variables (name->meaning), constraints, candidate_mathlib_lemmas, sources (url+title if present).",
                                                    "schema": extract_schema
                                                }
                                            ],
                                            "onlyMainContent": true
                                        }
                                    }
                                },
                                {
                                    "server": "user-tavily-remote-mcp",
                                    "toolName": "tavily_search",
                                    "arguments": {
                                        "query": web_q,
                                        "search_depth": "advanced",
                                        "max_results": 5
                                    }
                                },
                                {
                                    "server": "user-perplexity",
                                    "toolName": "search",
                                    "arguments": {
                                        "query": format!("Summarize the key lemma/step for: {q}. Extract variable definitions and constraints, and mention any standard references.")
                                    }
                                }
                                ],
                                "extract": {
                                    "schema": {
                                        "$ref": "#/definitions/extract_schema",
                                        "definitions": { "extract_schema": extract_schema }
                                    }
                                }
                            }
                        }
                    }));
                }
                // Warnings last: they are useful, but proof-blocking work is usually sorries/errors.
                for w in &warning_samples {
                    let lean_query = format!("Lean 4 {}", w);
                    next_actions.push(json!({
                        "kind": "fix_warning",
                        "file": file,
                        "message": w,
                        "research": {
                            "plan": {
                                "goal": "Decide whether this warning matters and how to silence it correctly.",
                                "calls": [
                                {
                                    "server": "user-arxiv-semantic-search-mcp",
                                    "toolName": "search_papers",
                                    "arguments": { "query": lean_query }
                                },
                                {
                                    "server": "user-tavily-remote-mcp",
                                    "toolName": "tavily_search",
                                    "arguments": {
                                        "query": lean_query,
                                        "search_depth": "basic",
                                        "max_results": 5
                                    }
                                }
                                ]
                            }
                        }
                    }));
                }

                table.push(json!({
                    "file": file,
                    "verify": {
                        "ok": ok,
                        "errors": errors,
                        "warnings": warnings,
                        "error_samples": error_samples,
                        "warning_samples": warning_samples,
                    },
                    "sorries": { "count": locs.len(), "conservative_count": conservative_sorries, "locations": locs },
                }));

                items.push(json!({
                    "file": file,
                    "verify": if include_raw_verify {
                        json!({ "raw": raw_v })
                    } else {
                        json!({ "raw": null })
                    },
                }));
            }

            let mut report_path_out: Option<String> = None;
            if let Some(out_path) = output_html {
                if let Some(parent) = out_path.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| format!("failed to create dir {}: {}", parent.display(), e))?;
                }

                let mut html = String::new();
                html.push_str("<!doctype html>\n<html><head><meta charset=\"utf-8\"/>\n");
                html.push_str("<title>proofpatch report</title>\n");
                html.push_str("<style>body{font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial;max-width:1200px;margin:24px auto;padding:0 16px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px;vertical-align:top}th{background:#f6f6f6;text-align:left}code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,\"Liberation Mono\",monospace}pre{white-space:pre-wrap}</style>\n");
                html.push_str("</head><body>\n");
                html.push_str("<h2>proofpatch report</h2>\n");
                html.push_str(&format!(
                    "<p><b>repo_root</b>: <code>{}</code></p>\n",
                    escape_html(&repo_root.display().to_string())
                ));
                html.push_str("<table>\n<thead><tr><th>file</th><th>verify</th><th>warnings (sample)</th><th>sorries</th></tr></thead>\n<tbody>\n");

                for row in &table {
                    let file = row.get("file").and_then(|v| v.as_str()).unwrap_or("");
                    let verify = row.get("verify").cloned().unwrap_or_else(|| json!({}));
                    let ok = verify.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
                    let errors = verify.get("errors").and_then(|v| v.as_u64()).unwrap_or(0);
                    let warnings = verify.get("warnings").and_then(|v| v.as_u64()).unwrap_or(0);
                    let warning_samples = verify
                        .get("warning_samples")
                        .and_then(|v| v.as_array())
                        .cloned()
                        .unwrap_or_default();
                    let sorries = row
                        .get("sorries")
                        .and_then(|v| v.get("locations"))
                        .and_then(|v| v.as_array())
                        .cloned()
                        .unwrap_or_default();

                    html.push_str("<tr>");
                    html.push_str(&format!("<td><code>{}</code></td>", escape_html(file)));
                    html.push_str(&format!(
                        "<td><b>ok</b>: {}<br/><b>errors</b>: {}<br/><b>warnings</b>: {}</td>",
                        ok, errors, warnings
                    ));
                    if !warning_samples.is_empty() {
                        html.push_str("<td><b>warnings (sample)</b><pre>");
                        for w in warning_samples.iter().take(6) {
                            let s = w.as_str().unwrap_or("");
                            html.push_str(&escape_html(s));
                            html.push('\n');
                        }
                        html.push_str("</pre></td>");
                    } else {
                        html.push_str("<td><b>warnings (sample)</b><pre>(none)</pre></td>");
                    }
                    html.push_str("<td>");
                    html.push_str(&format!("<b>count</b>: {}<br/>", sorries.len()));
                    for loc in sorries.iter().take(8) {
                        let token = loc.get("token").and_then(|v| v.as_str()).unwrap_or("sorry");
                        let decl_kind = loc.get("decl_kind").and_then(|v| v.as_str()).unwrap_or("");
                        let decl_name = loc.get("decl_name").and_then(|v| v.as_str()).unwrap_or("");
                        let decl_label = if !decl_kind.is_empty() && !decl_name.is_empty() {
                            format!("{} {}", decl_kind, decl_name)
                        } else {
                            "".to_string()
                        };
                        let line = loc.get("line").and_then(|v| v.as_u64()).unwrap_or(0);
                        let col = loc.get("col").and_then(|v| v.as_u64()).unwrap_or(0);
                        let excerpt = loc.get("excerpt").and_then(|v| v.as_str()).unwrap_or("");
                        html.push_str(&format!(
                            "<div><b>@</b> {}:{} <code>{}</code> <code>{}</code><pre>{}</pre></div>",
                            line,
                            col,
                            escape_html(token),
                            escape_html(&decl_label),
                            escape_html(excerpt)
                        ));
                    }
                    html.push_str("</td>");
                    html.push_str("</tr>\n");
                }

                html.push_str("</tbody></table></body></html>\n");
                std::fs::write(&out_path, html.as_bytes())
                    .map_err(|e| format!("failed to write html {}: {e}", out_path.display()))?;
                report_path_out = Some(out_path.display().to_string());
            }

            let out = json!({
                "repo_root": repo_root.display().to_string(),
                "table": table,
                "next_actions": next_actions,
                "html_path": report_path_out,
            });
            println!("{}", out.to_string());
            Ok(())
        }

        "context-pack" => {
            let repo_root = arg_value(rest, "--repo")
                .ok_or_else(|| "missing --repo".to_string())
                .map(PathBuf::from)?;
            let file = arg_value(rest, "--file").ok_or_else(|| "missing --file".to_string())?;
            let decl = arg_value(rest, "--decl");
            let line = arg_u64(rest, "--line").map(|x| x as usize);
            let context_lines = arg_u64(rest, "--context-lines").unwrap_or(25) as usize;
            let nearby_lines = arg_u64(rest, "--nearby-lines").unwrap_or(80) as usize;
            let max_nearby = arg_u64(rest, "--max-nearby").unwrap_or(30) as usize;
            let max_imports = arg_u64(rest, "--max-imports").unwrap_or(50) as usize;
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);

            let repo_root =
                plc::find_lean_repo_root(&repo_root).map_err(|e| format!("repo_root: {e}"))?;
            let pack = plc::build_context_pack(
                &repo_root,
                &file,
                decl.as_deref(),
                line,
                context_lines,
                nearby_lines,
                max_nearby,
                max_imports,
            )?;
            let out = serde_json::to_value(pack)
                .map_err(|e| format!("failed to serialize context pack: {e}"))?;
            if let Some(p) = output_json {
                write_json(&p, &out)?;
                println!(
                    "{}",
                    json!({
                        "ok": true,
                        "written": p.display().to_string(),
                        "kind": "context_pack",
                        "result_kind": serde_json::Value::Null,
                    })
                    .to_string()
                );
            } else {
                println!("{}", out.to_string());
            }
            Ok(())
        }

        "scratch-lemma" => {
            let repo_root = arg_value(rest, "--repo")
                .ok_or_else(|| "missing --repo".to_string())
                .map(PathBuf::from)?;
            let file = arg_value(rest, "--file").ok_or_else(|| "missing --file".to_string())?;
            let name = arg_value(rest, "--name").ok_or_else(|| "missing --name".to_string())?;
            let kind = arg_value(rest, "--kind").unwrap_or_else(|| "theorem".to_string());
            if kind != "theorem" && kind != "lemma" {
                return Err("--kind must be theorem or lemma".to_string());
            }
            let out_rel = arg_value(rest, "--out")
                .unwrap_or_else(|| format!(".generated/{}_Scratch.lean", name.replace('.', "_")));

            let repo_root =
                plc::find_lean_repo_root(&repo_root).map_err(|e| format!("repo_root: {e}"))?;

            // Minimal, reliable scratch template for “work backwards” iterations.
            // `file` can be either:
            // - a `.lean` path (e.g. `GeometryOfNumbers/Computable/LLLExactTermination.lean`), or
            // - an import module path (e.g. `GeometryOfNumbers.Computable.LLLExactTermination`).
            let import_mod = if file.ends_with(".lean") || file.contains('/') || file.contains('\\')
            {
                file.trim_end_matches(".lean")
                    .replace(['/', '\\'], ".")
                    .trim_matches('.')
                    .to_string()
            } else {
                file.trim_matches('.').to_string()
            };
            let mut txt = String::new();
            txt.push_str(&format!("import {import_mod}\n\n"));
            txt.push_str("namespace ProofpatchScratch\n\n");
            txt.push_str("/-- Scratch lemma scaffold.\n\n");
            txt.push_str("Replace the statement and proof, but keep a `sorry` while iterating.\n");
            txt.push_str("-/\n");
            txt.push_str(&format!("{kind} {name}_scratch : True := by\n"));
            txt.push_str("  sorry\n\n");
            txt.push_str("end ProofpatchScratch\n");

            let out_abs = repo_root.join(&out_rel);
            if let Some(parent) = out_abs.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("failed to create dir {}: {e}", parent.display()))?;
            }
            std::fs::write(&out_abs, txt.as_bytes())
                .map_err(|e| format!("write {}: {e}", out_abs.display()))?;

            println!(
                "{}",
                json!({
                    "ok": true,
                    "kind": "scratch_lemma",
                    "repo_root": repo_root.display().to_string(),
                    "source_file": file,
                    "import_mod": import_mod,
                    "name": name,
                    "out_rel": out_rel,
                    "out_abs": out_abs.display().to_string(),
                })
                .to_string()
            );
            Ok(())
        }

        "arxiv-search" => {
            let query = arg_value(rest, "--query").ok_or_else(|| "missing --query".to_string())?;
            let max_results = arg_u64(rest, "--max-results").unwrap_or(8).clamp(1, 50) as usize;
            let timeout_ms = arg_u64(rest, "--timeout-ms").unwrap_or(20_000);
            let must_include = arg_values(rest, "--must-include");
            let must_include_all = arg_values(rest, "--must-include-all");
            let llm_summary = arg_flag(rest, "--llm-summary");
            let llm_timeout_s = arg_u64(rest, "--llm-timeout-s").unwrap_or(20);
            let quiet = arg_flag(rest, "--quiet");
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);

            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| format!("failed to build tokio runtime: {e}"))?;

            let papers = rt.block_on(plc::arxiv::arxiv_search(
                &query,
                max_results,
                StdDuration::from_millis(timeout_ms),
            ))?;

            // Post-filtering: ArXiv search can return "SWAP" papers when the query contains "swap".
            // If the user didn't specify a filter but the query looks LLL-ish, filter automatically.
            let mut must: Vec<String> = must_include;
            let ql = query.to_lowercase();
            if must.is_empty()
                && (ql.contains("lll") || ql.contains("lenstra") || ql.contains("lovasz"))
            {
                must = vec![
                    "lll".into(),
                    "lenstra".into(),
                    "lovasz".into(),
                    "lattice".into(),
                ];
            }
            let papers: Vec<plc::arxiv::ArxivPaper> = {
                let must_any_l: Vec<String> = must.iter().map(|s| s.to_lowercase()).collect();
                let must_all_l: Vec<String> =
                    must_include_all.iter().map(|s| s.to_lowercase()).collect();
                if must_any_l.is_empty() && must_all_l.is_empty() {
                    papers
                } else {
                    papers
                        .into_iter()
                        .filter(|p| {
                            let hay = format!("{}\n{}", p.title, p.abstract_text).to_lowercase();
                            let ok_any = must_any_l.is_empty()
                                || must_any_l.iter().any(|tok| hay.contains(tok));
                            let ok_all = must_all_l.is_empty()
                                || must_all_l.iter().all(|tok| hay.contains(tok));
                            ok_any && ok_all
                        })
                        .collect()
                }
            };

            let mut out = json!({
                "ok": true,
                "kind": "arxiv_search",
                "query": query,
                "max_results": max_results,
                "filter": { "must_include_any": must, "must_include_all": must_include_all },
                "papers": papers,
                // A minimal envelope compatible with `research-ingest`.
                "research": {
                    "tool": "arxiv",
                    "papers": papers.iter().map(|p| json!({
                        "title": p.title,
                        // Use a canonical `url` key so downstream ingestion reliably captures snippets.
                        "url": p.link,
                        "link": p.link,
                        "pdf_url": p.pdf_url,
                        "abstract": p.abstract_text,
                        "snippet": truncate_str(&p.abstract_text, 400),
                        "authors": p.authors,
                        "published": p.published,
                        "updated": p.updated,
                    })).collect::<Vec<_>>()
                }
            });

            if llm_summary {
                // Ensure the LLM can find keys from repo/.env or the dev/.env super-workspace.
                if let Ok(cwd) = std::env::current_dir() {
                    if let Ok(git_root) = plc::review::git_repo_root(&cwd) {
                        plc::load_dotenv_smart(&git_root);
                    } else {
                        plc::load_dotenv_smart(&cwd);
                    }
                }
                let kind = research_summary_kind_default();
                let system = research_summary_system_prompt(kind);
                let user = serde_json::to_string(&json!({"query": query, "papers": papers}))
                    .unwrap_or_else(|_| "{\"papers\":[]}".to_string());
                let res = rt.block_on(plc::llm::chat_completion_structured::<ResearchSummary>(
                    &system,
                    &user,
                    StdDuration::from_secs(llm_timeout_s),
                ));
                match res {
                    Ok(r) => {
                        let v = serde_json::to_value(&r.value)
                            .unwrap_or_else(|_| serde_json::Value::Null);
                        out["llm_summary"] = json!({
                            "provider": r.provider,
                            "model": r.model,
                            "model_source": r.model_source,
                            "model_env": r.model_env,
                            "mode": r.mode,
                            "kind": kind,
                            "content": serde_json::to_string(&r.value).unwrap_or_default(),
                            "content_struct": v,
                            "raw": r.raw
                        });
                    }
                    Err(e) => {
                        out["llm_summary"] = json!({"ok": false, "error": e});
                    }
                }
            }

            if !quiet {
                let titles: Vec<String> = papers
                    .iter()
                    .take(3)
                    .map(|p| truncate_str(&p.title, 90))
                    .collect();
                eprintln!(
                    "arxiv-search: results={} query=\"{}\" top_titles={}",
                    papers.len(),
                    truncate_str(&query, 120),
                    serde_json::to_string(&titles).unwrap_or_else(|_| "[]".to_string())
                );
            }

            if let Some(p) = output_json {
                write_json(&p, &out)?;
                println!(
                    "{}",
                    json!({
                        "ok": true,
                        "written": p.display().to_string(),
                        "kind": "arxiv_search",
                        "result_kind": serde_json::Value::Null,
                    })
                    .to_string()
                );
            } else {
                println!("{}", out.to_string());
            }
            Ok(())
        }

        "research-auto" => {
            let repo_root = arg_value(rest, "--repo")
                .ok_or_else(|| "missing --repo".to_string())
                .map(PathBuf::from)?;
            let preset_name =
                arg_value(rest, "--preset").ok_or_else(|| "missing --preset".to_string())?;
            let quiet = arg_flag(rest, "--quiet");
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);

            let repo_root =
                plc::find_lean_repo_root(&repo_root).map_err(|e| format!("repo_root: {e}"))?;

            // Ensure keys are visible from either <repo>/.env or dev/.env.
            plc::load_dotenv_smart(&repo_root);

            let cfg = plc::config::load_from_repo_root(&repo_root)?.ok_or_else(|| {
                format!(
                    "missing config: {}",
                    plc::config::config_path(&repo_root).display()
                )
            })?;
            let preset = cfg.research.resolve_preset(&preset_name).ok_or_else(|| {
                let mut names: Vec<String> = cfg.research.presets.keys().cloned().collect();
                names.sort();
                format!(
                    "unknown preset: {} (available: {})",
                    preset_name,
                    names.join(", ")
                )
            })?;

            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| format!("failed to build tokio runtime: {e}"))?;

            // Best-effort: if arXiv fetch fails, still write a structured artifact when
            // `--output-json` is set (and return a non-zero exit via `Err(...)`).
            let arxiv_res = rt.block_on(plc::arxiv::arxiv_search(
                &preset.query,
                preset.max_results,
                StdDuration::from_millis(preset.timeout_ms),
            ));
            let (mut papers, arxiv_error): (Vec<plc::arxiv::ArxivPaper>, Option<String>) =
                match arxiv_res {
                    Ok(p) => (p, None),
                    Err(e) => (vec![], Some(format!("{e}"))),
                };

            if !preset.must_include_any.is_empty() || !preset.must_include_all.is_empty() {
                let must_any_l: Vec<String> = preset
                    .must_include_any
                    .iter()
                    .map(|s| s.to_lowercase())
                    .collect();
                let must_all_l: Vec<String> = preset
                    .must_include_all
                    .iter()
                    .map(|s| s.to_lowercase())
                    .collect();
                papers = papers
                    .into_iter()
                    .filter(|p| {
                        let hay = format!("{}\n{}", p.title, p.abstract_text).to_lowercase();
                        let ok_any =
                            must_any_l.is_empty() || must_any_l.iter().any(|tok| hay.contains(tok));
                        let ok_all =
                            must_all_l.is_empty() || must_all_l.iter().all(|tok| hay.contains(tok));
                        ok_any && ok_all
                    })
                    .collect();
            }

            let mut out = json!({
                "ok": arxiv_error.is_none(),
                "kind": "research_auto",
                "repo_root": repo_root.display().to_string(),
                "config_path": plc::config::config_path(&repo_root).display().to_string(),
                "preset": preset_name,
                "settings": preset,
                "arxiv": {
                    "ok": arxiv_error.is_none(),
                    "error": arxiv_error,
                    "kind": "arxiv_search",
                    "query": preset.query,
                    "max_results": preset.max_results,
                    "filter": { "must_include_any": preset.must_include_any, "must_include_all": preset.must_include_all },
                    "papers": papers,
                    "research": {
                        "tool": "arxiv",
                        "papers": papers.iter().map(|p| json!({
                            "title": p.title,
                            "url": p.link,
                            "link": p.link,
                            "pdf_url": p.pdf_url,
                            "abstract": p.abstract_text,
                            "snippet": truncate_str(&p.abstract_text, 400),
                            "authors": p.authors,
                            "published": p.published,
                            "updated": p.updated,
                        })).collect::<Vec<_>>()
                    }
                }
            });

            // Only attempt an LLM summary if we successfully fetched papers.
            if preset.llm_summary && out["arxiv"]["ok"].as_bool().unwrap_or(false) {
                let kind = preset
                    .llm_summary_kind
                    .as_deref()
                    .unwrap_or(research_summary_kind_default());
                let system = research_summary_system_prompt(kind);
                let user = serde_json::to_string(&json!({
                    "preset": preset_name,
                    "query": out["arxiv"]["query"],
                    "papers": out["arxiv"]["papers"],
                }))
                .unwrap_or_else(|_| "{\"papers\":[]}".to_string());
                let res = match normalize_summary_kind(kind).as_str() {
                    "formalization_v2" => rt
                        .block_on(plc::llm::chat_completion_structured::<ResearchSummaryV2>(
                            &system,
                            &user,
                            StdDuration::from_secs(preset.llm_timeout_s),
                        ))
                        .map(|r| {
                            let capped = cap_summary_v2(r.value, &preset);
                            // Erase the type into JSON for downstream consumption.
                            let v =
                                serde_json::to_value(&capped).unwrap_or(serde_json::Value::Null);
                            (
                                r.provider,
                                r.model,
                                r.model_source,
                                r.model_env,
                                r.mode,
                                v,
                                r.raw,
                            )
                        }),
                    _ => rt
                        .block_on(plc::llm::chat_completion_structured::<ResearchSummary>(
                            &system,
                            &user,
                            StdDuration::from_secs(preset.llm_timeout_s),
                        ))
                        .map(|r| {
                            let capped = cap_summary_v1(r.value, &preset);
                            let v =
                                serde_json::to_value(&capped).unwrap_or(serde_json::Value::Null);
                            (
                                r.provider,
                                r.model,
                                r.model_source,
                                r.model_env,
                                r.mode,
                                v,
                                r.raw,
                            )
                        }),
                };
                match res {
                    Ok((provider, model, model_source, model_env, mode, v, raw)) => {
                        out["arxiv"]["llm_summary"] = json!({
                            "provider": provider,
                            "model": model,
                            "model_source": model_source,
                            "model_env": model_env,
                            "mode": mode,
                            "kind": kind,
                            "content": v.to_string(),
                            "content_struct": v,
                            "raw": raw
                        });
                    }
                    Err(e) => {
                        out["arxiv"]["llm_summary"] = json!({"ok": false, "error": e});
                    }
                }
            }

            // Emit a ready-to-consume note bundle for `research-attach`.
            let notes = plc::ingest_research_json(&out);
            out["research_notes"] = serde_json::to_value(notes)
                .map_err(|e| format!("failed to serialize research notes: {e}"))?;

            if !quiet {
                let n = out["research_notes"]["deduped_urls"].as_u64().unwrap_or(0);
                eprintln!(
                    "research-auto: preset={} urls={} repo={}",
                    preset_name,
                    n,
                    repo_root.display()
                );
            }

            if let Some(p) = output_json {
                write_json(&p, &out)?;
                println!(
                    "{}",
                    json!({
                        "ok": out["ok"].as_bool().unwrap_or(false),
                        "written": p.display().to_string(),
                        "error": out["arxiv"]["error"],
                    })
                    .to_string()
                );
            } else {
                println!("{}", out.to_string());
            }
            if out["ok"].as_bool().unwrap_or(false) {
                Ok(())
            } else {
                Err(format!(
                    "arxiv fetch: {}",
                    out["arxiv"]["error"].as_str().unwrap_or("unknown error")
                ))
            }
        }

        "research-ingest" => {
            let input = arg_value(rest, "--input")
                .ok_or_else(|| "missing --input".to_string())
                .map(PathBuf::from)?;
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);

            let txt = std::fs::read_to_string(&input)
                .map_err(|e| format!("read {}: {e}", input.display()))?;
            let v: serde_json::Value =
                serde_json::from_str(&txt).map_err(|e| format!("json parse: {e}"))?;
            let notes = plc::ingest_research_json(&v);
            let out = serde_json::to_value(notes).map_err(|e| format!("json encode: {e}"))?;

            if let Some(p) = output_json {
                write_json(&p, &out)?;
                println!(
                    "{}",
                    json!({
                        "ok": true,
                        "written": p.display().to_string(),
                        "kind": "research_ingest",
                        "result_kind": serde_json::Value::Null,
                    })
                    .to_string()
                );
            } else {
                println!("{}", out.to_string());
            }
            Ok(())
        }

        "research-attach" => {
            let report_path = arg_value(rest, "--report-json")
                .ok_or_else(|| "missing --report-json".to_string())
                .map(PathBuf::from)?;
            let notes_path = arg_value(rest, "--research-notes")
                .ok_or_else(|| "missing --research-notes".to_string())
                .map(PathBuf::from)?;
            let top_k = arg_u64(rest, "--top-k").unwrap_or(3) as usize;
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);

            let report_txt = std::fs::read_to_string(&report_path)
                .map_err(|e| format!("read {}: {e}", report_path.display()))?;
            let mut report_v: serde_json::Value =
                serde_json::from_str(&report_txt).map_err(|e| format!("json parse: {e}"))?;

            let notes_txt = std::fs::read_to_string(&notes_path)
                .map_err(|e| format!("read {}: {e}", notes_path.display()))?;
            let notes: plc::ResearchNotes =
                serde_json::from_str(&notes_txt).map_err(|e| format!("json parse: {e}"))?;

            plc::attach_research_matches_to_next_actions(&mut report_v, &notes, top_k);
            let out = report_v;

            if let Some(p) = output_json {
                write_json(&p, &out)?;
                println!(
                    "{}",
                    json!({
                        "ok": true,
                        "written": p.display().to_string(),
                        "kind": "research_attach",
                        "result_kind": serde_json::Value::Null,
                    })
                    .to_string()
                );
            } else {
                println!("{}", out.to_string());
            }
            Ok(())
        }

        _ => Err(usage()),
    }
}
