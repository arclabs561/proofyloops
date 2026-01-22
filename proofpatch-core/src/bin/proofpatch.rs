#![recursion_limit = "256"]

use proofpatch_core as plc;
use serde_json::json;
use schemars::schema_for;
use similar::TextDiff;
use std::path::PathBuf;
use std::time::Duration as StdDuration;
use std::{fs, io};
use durability::storage::Directory as _; // for `atomic_write`
use durability::FsDirectory;

fn extract_json_from_text(s: &str) -> Option<serde_json::Value> {
    // 1) Prefer fenced ```json ... ``` blocks.
    if let Some(i) = s.find("```json") {
        let rest = &s[i + "```json".len()..];
        if let Some(j) = rest.find("```") {
            let cand = rest[..j].trim();
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(cand) {
                return Some(v);
            }
        }
    }
    // 2) Fall back to first {...} span.
    let i = s.find('{')?;
    let j = s.rfind('}')?;
    if j <= i {
        return None;
    }
    let cand = s[i..=j].trim();
    serde_json::from_str::<serde_json::Value>(cand).ok()
}

fn default_context_lines() -> u64 {
    8
}
fn default_nearby_lines() -> u64 {
    120
}
fn default_max_nearby() -> u64 {
    30
}
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

fn env_reasoning_request() -> Option<axi::ReasoningRequest> {
    // Best-effort env wiring for “extra reasoning”.
    //
    // This is intentionally conservative: if the env vars are absent or invalid,
    // we default to `None` (no special request).
    //
    // Supported env vars:
    // - PROOFPATCH_REASONING_EFFORT = low|medium|high
    // - PROOFPATCH_REASONING_BUDGET_TOKENS = <usize>
    let effort = std::env::var("PROOFPATCH_REASONING_EFFORT")
        .ok()
        .and_then(|s| match s.trim().to_ascii_lowercase().as_str() {
            "low" => Some(axi::ReasoningEffort::Low),
            "medium" => Some(axi::ReasoningEffort::Medium),
            "high" => Some(axi::ReasoningEffort::High),
            _ => None,
        });

    let budget_tokens = std::env::var("PROOFPATCH_REASONING_BUDGET_TOKENS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok());

    if effort.is_none() && budget_tokens.is_none() {
        return None;
    }
    Some(axi::ReasoningRequest {
        effort,
        budget_tokens,
    })
}

fn default_agent_reasoning_request(_provider: &str) -> Option<axi::ReasoningRequest> {
    // Internal defaults (conservative):
    //
    // When tool calling is enabled, some provider routes (notably OpenRouter → Anthropic)
    // can become less reliable if “thinking” is enabled, due to strict content block
    // sequencing requirements around tool_use/tool_result.
    //
    // So: default to *no* explicit reasoning request for agent/tool runs.
    None
}

fn effective_agent_reasoning_request(provider: &str) -> Option<axi::ReasoningRequest> {
    // Env is an advanced override; otherwise fall back to internal default.
    env_reasoning_request().or_else(|| default_agent_reasoning_request(provider))
}

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

fn default_timeout_s() -> u64 {
    90
}

#[derive(Debug, Clone, serde::Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(deny_unknown_fields)]
struct VerifySummaryArgs {
    file: String,
    #[schemars(required)]
    #[serde(default = "default_timeout_s")]
    timeout_s: u64,
}

fn default_sorry_context_lines() -> u64 {
    1
}

fn default_max_sorries() -> u64 {
    30
}

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

fn default_patch_timeout_s() -> u64 {
    120
}

fn default_patch_verify() -> bool {
    true
}

fn default_patch_write() -> bool {
    false
}

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

fn read_json(path: &std::path::Path) -> Option<serde_json::Value> {
    let s = std::fs::read_to_string(path).ok()?;
    serde_json::from_str::<serde_json::Value>(&s).ok()
}

fn durable_atomic_write(cache_root: &std::path::Path, rel: &str, data: &[u8]) {
    // Best-effort: durability provides atomic write + sync barriers on fs backends.
    // If it fails (permissions, etc.), fall back to a plain write.
    if let Ok(dir) = FsDirectory::new(cache_root) {
        let _ = dir.atomic_write(rel, data);
    } else {
        let p = cache_root.join(rel);
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(p, data);
    }
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

#[cfg(feature = "smt")]
fn cache_read_smt_entails(cache_dir: &std::path::Path, state_key: u64, goal_sig: u64) -> Option<bool> {
    let p = cache_dir
        .join("smt")
        .join(format!("{state_key}_{goal_sig}.json"));
    let v = read_json(&p)?;
    v.get("entails").and_then(|x| x.as_bool())
}

#[cfg(feature = "smt")]
fn cache_write_smt_entails(cache_dir: &std::path::Path, state_key: u64, goal_sig: u64, entails: bool) {
    let rel = format!("smt/{state_key}_{goal_sig}.json");
    let v = json!({ "state_key": state_key, "goal_sig": goal_sig, "entails": entails });
    durable_atomic_write(cache_dir, &rel, v.to_string().as_bytes());
}

#[cfg(feature = "smt")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SmtVarKind {
    Int,
    Nat,
}

#[cfg(feature = "smt")]
fn smt_sanitize_name(s: &str) -> String {
    let mut out = String::new();
    for ch in s.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out = "x".to_string();
    }
    // SMT-LIB symbols cannot start with a digit unless quoted; keep it simple.
    if out.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
        out.insert(0, '_');
    }
    out
}

#[cfg(feature = "smt")]
fn smt_extract_decl_kind(hyp_text: &str) -> Option<(String, SmtVarKind)> {
    // Very small recognizer for declarations like:
    // - `n : ℕ`
    // - `k : Nat`
    // - `m : ℤ`
    // - `x : Int`
    let (name, ty) = hyp_text.split_once(':')?;
    let name = name.trim();
    let ty = ty.trim();
    if name.is_empty() || ty.is_empty() {
        return None;
    }
    let kind = if ty.contains('ℕ') || ty.contains("Nat") {
        SmtVarKind::Nat
    } else if ty.contains('ℤ') || ty.contains("Int") {
        SmtVarKind::Int
    } else {
        return None;
    };
    Some((smt_sanitize_name(name), kind))
}

#[cfg(feature = "smt")]
#[derive(Debug, Clone)]
struct LinearExpr {
    // var -> coefficient
    coeffs: std::collections::BTreeMap<String, i64>,
    c0: i64,
}

#[cfg(feature = "smt")]
fn parse_linear_expr_int(s: &str) -> Option<LinearExpr> {
    // Extremely small parser: supports sums/differences of identifiers and integer literals.
    // Rejects anything with obvious non-LIA operators (*,/ ,^,·, etc).
    let bad = ['*', '/', '^', '·', '↑', '∑', '∏'];
    if s.chars().any(|c| bad.contains(&c)) {
        return None;
    }
    let mut coeffs: std::collections::BTreeMap<String, i64> = std::collections::BTreeMap::new();
    let mut c0: i64 = 0;
    let mut i = 0usize;
    let chars: Vec<char> = s.chars().collect();
    let mut sign: i64 = 1;
    while i < chars.len() {
        let ch = chars[i];
        if ch.is_whitespace() {
            i += 1;
            continue;
        }
        if ch == '+' {
            sign = 1;
            i += 1;
            continue;
        }
        if ch == '-' {
            sign = -1;
            i += 1;
            continue;
        }
        // integer literal
        if ch.is_ascii_digit() {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            let lit: String = chars[i..j].iter().collect();
            let v: i64 = lit.parse().ok()?;
            c0 = c0.saturating_add(sign.saturating_mul(v));
            i = j;
            continue;
        }
        // identifier (unicode alnum + '_' + '.' treated as part; we sanitize later)
        if ch.is_alphanumeric() || ch == '_' || ch == '.' {
            let mut j = i + 1;
            while j < chars.len() && (chars[j].is_alphanumeric() || chars[j] == '_' || chars[j] == '.') {
                j += 1;
            }
            let raw: String = chars[i..j].iter().collect();
            let name = smt_sanitize_name(&raw);
            *coeffs.entry(name).or_insert(0) = coeffs
                .get(&name)
                .copied()
                .unwrap_or(0)
                .saturating_add(sign);
            i = j;
            continue;
        }
        // Unknown token.
        return None;
    }
    Some(LinearExpr { coeffs, c0 })
}

#[cfg(feature = "smt")]
fn linear_expr_to_smt_sexp(e: &LinearExpr) -> smtkit::sexp::Sexp {
    use smtkit::smt2::t;
    let mut terms: Vec<smtkit::sexp::Sexp> = Vec::new();
    if e.c0 != 0 {
        terms.push(t::int_lit(e.c0));
    }
    for (v, c) in e.coeffs.iter() {
        if *c == 0 {
            continue;
        }
        let sym = t::sym(v.clone());
        if *c == 1 {
            terms.push(sym);
        } else if *c == -1 {
            terms.push(t::app("-", vec![sym]));
        } else {
            terms.push(t::app("*", vec![t::int_lit(*c), sym]));
        }
    }
    if terms.is_empty() {
        t::int_lit(0)
    } else if terms.len() == 1 {
        terms[0].clone()
    } else {
        t::add(terms)
    }
}

#[cfg(feature = "smt")]
#[derive(Debug, Clone)]
struct ParsedRelConstraint {
    sexp: smtkit::sexp::Sexp,
    vars: std::collections::BTreeSet<String>,
}

#[cfg(feature = "smt")]
fn parse_rel_constraint_int(s: &str) -> Option<ParsedRelConstraint> {
    // Find a single top-level relation and parse both sides as linear integer expressions.
    // Supports ASCII and unicode relations: <=, ≥, ≤, >=, <, >, =.
    let s = s.trim();
    let ops = ["<=", "≤", ">=", "≥", "<", ">", "="];
    let (op, idx) = ops.iter().find_map(|op| s.find(op).map(|i| (*op, i)))?;
    let (lhs, rhs0) = s.split_at(idx);
    let rhs = rhs0.get(op.len()..)?;
    let lhs_e = parse_linear_expr_int(lhs.trim())?;
    let rhs_e = parse_linear_expr_int(rhs.trim())?;
    use smtkit::smt2::t;
    let a = linear_expr_to_smt_sexp(&lhs_e);
    let b = linear_expr_to_smt_sexp(&rhs_e);
    let sexp = match op {
        "<=" | "≤" => t::le(a, b),
        ">=" | "≥" => t::ge(a, b),
        "<" => t::lt(a, b),
        ">" => t::app(">", vec![a, b]),
        "=" => t::eq(a, b),
        _ => return None,
    };
    let mut vars: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    vars.extend(lhs_e.coeffs.keys().cloned());
    vars.extend(rhs_e.coeffs.keys().cloned());
    Some(ParsedRelConstraint { sexp, vars })
}

#[cfg(feature = "smt")]
fn smt_entails_from_pp_dump(
    pp_dump: &serde_json::Value,
    timeout_ms: u64,
    seed: u64,
) -> Result<Option<bool>, String> {
    use smtkit::smt2::t;

    let goal = pp_dump
        .get("goals")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .ok_or_else(|| "pp_dump missing goals[0]".to_string())?;

    let pretty = goal
        .get("pretty")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let target = pretty
        .lines()
        .find_map(|ln| ln.trim_start().strip_prefix("⊢").map(|r| r.trim().to_string()))
        .unwrap_or_default();
    if target.is_empty() {
        return Ok(None);
    }

    // Collect variable kinds from hypothesis declaration lines.
    //
    // This is a deliberately *conservative* guardrail: if we cannot confidently classify
    // all variables as Int/Nat, we return `Ok(None)` rather than risk an unsound “entailed”.
    let mut var_kinds: std::collections::BTreeMap<String, SmtVarKind> = std::collections::BTreeMap::new();
    if let Some(hyps) = goal.get("hyps").and_then(|v| v.as_array()) {
        for h in hyps {
            if let Some(txt) = h.get("text").and_then(|v| v.as_str()) {
                if let Some((name, kind)) = smt_extract_decl_kind(txt) {
                    var_kinds.insert(name, kind);
                }
            }
        }
    }
    // Only proceed if the target itself parses as an int relation.
    let target_rel = match parse_rel_constraint_int(&target) {
        Some(r) => r,
        None => return Ok(None),
    };

    // Parse any hypothesis constraints we can.
    let mut hyp_rels: Vec<ParsedRelConstraint> = Vec::new();
    if let Some(hyps) = goal.get("hyps").and_then(|v| v.as_array()) {
        for h in hyps {
            if let Some(txt) = h.get("text").and_then(|v| v.as_str()) {
                // take substring after `:`
                let rhs = txt.split_once(':').map(|(_, r)| r.trim()).unwrap_or("");
                if rhs.is_empty() {
                    continue;
                }
                if let Some(r) = parse_rel_constraint_int(rhs) {
                    hyp_rels.push(r);
                }
            }
        }
    }

    // Ensure all mentioned vars have known kinds.
    // If a variable slips through with an unknown sort, we abort. This keeps the SMT hint as a
    // safe heuristic (it might be missing, but it shouldn't be wrong for type/sort reasons).
    for m in target_rel.vars.iter().chain(hyp_rels.iter().flat_map(|r| r.vars.iter())) {
        if !var_kinds.contains_key(m) {
            // Unknown sort; abort rather than risk unsound “entails”.
            return Ok(None);
        }
    }

    let (mut sess, _used) = match smtkit::session::spawn_auto() {
        Ok(v) => v,
        Err(_e) => return Ok(None), // no solver available
    };
    sess.set_logic("QF_LIA").map_err(|e| e.to_string())?;
    sess.set_print_success(false).map_err(|e| e.to_string())?;
    sess.set_produce_models(false).map_err(|e| e.to_string())?;
    sess.set_timeout_ms(timeout_ms).map_err(|e| e.to_string())?;
    sess.set_random_seed(seed).map_err(|e| e.to_string())?;

    // Declare vars as Int. Add Nat non-negativity constraints.
    for (name, kind) in var_kinds.iter() {
        sess.declare_const(name, &smtkit::smt2::Sort::Int.to_smt2())
            .map_err(|e| e.to_string())?;
        if *kind == SmtVarKind::Nat {
            sess.assert_sexp(&t::ge(t::sym(name.clone()), t::int_lit(0)))
                .map_err(|e| e.to_string())?;
        }
    }
    for r in hyp_rels {
        sess.assert_sexp(&r.sexp).map_err(|e| e.to_string())?;
    }
    // Check satisfiability of hyps ∧ ¬target.
    //
    // Interpretation:
    // - UNSAT ⇒ hyps ⇒ target  (within our parsed LIA fragment)
    // - SAT   ⇒ hyps ⊬ target  (found countermodel in LIA)
    // - UNKNOWN ⇒ do not use as a signal
    sess.assert_sexp(&t::not(target_rel.sexp)).map_err(|e| e.to_string())?;
    let st = sess.check_sat().map_err(|e| e.to_string())?;
    match st {
        smtkit::session::Status::Unsat => Ok(Some(true)),
        smtkit::session::Status::Sat => Ok(Some(false)),
        smtkit::session::Status::Unknown => Ok(None),
    }
}

#[cfg(feature = "smt")]
fn smt_entails_from_hyps_target(
    hyps_texts: &[String],
    target: &str,
    timeout_ms: u64,
    seed: u64,
) -> Result<Option<bool>, String> {
    // Build a minimal pp_dump-shaped object so we can reuse the same logic.
    //
    // This must stay “no extra Lean calls”: the input is cached hypothesis text + a target string.
    // Any failure to parse/classify sorts should return `Ok(None)`.
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
    smt_entails_from_pp_dump(&pp_dump, timeout_ms, seed)
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
        "proofpatch — direct CLI (no MCP).",
        "",
        "Commands:",
        "  triage-file --repo <path> --file <relpath> [--timeout-s N] [--max-sorries N] [--context-lines N] [--include-raw-verify] [--no-context-pack] [--no-prompts] [--output-json <path>]",
        "  locate-sorries --repo <path> --file <relpath> [--max-sorries N] [--context-lines N] [--output-json <path>]",
        "  lean-embed-smoke  (requires cargo feature `lean-embed`)",
        "  verify-summary --repo <path> --file <relpath> [--timeout-s N] [--include-raw-verify] [--output-json <path>]",
        "  agent-step  --repo <path> --file <relpath> [--timeout-s N] [--write] [--output-json <path>]",
        "  prompt      --repo <path> --file <relpath> --lemma <name> [--output-json <path>]",
        "  rubberduck-prompt --repo <path> --file <relpath> --lemma <name> [--diagnostics-file <path>] [--output-json <path>]",
        "  patch       --repo <path> --file <relpath> --lemma <name> --replacement-file <path> [--timeout-s N] [--write] [--include-raw-verify] [--output-json <path>]",
        "  patch-region --repo <path> --file <relpath> --start-line N --end-line N --replacement-file <path> [--timeout-s N] [--write] [--include-raw-verify] [--output-json <path>]",
        "  patch-nearest --repo <path> --file <relpath> --replacement-file <path> [--timeout-s N] [--write] [--max-sorries N] [--context-lines N] [--include-raw-verify] [--output-json <path>]",
        "  tree-search-nearest --repo <path> --file <relpath> [--timeout-s N] [--total-timeout-s N] [--log-level 0|1|2] [--events-jsonl <path>] [--events-keep N] [--events-all-keep N] [--beam N] [--max-nodes N] [--depth N] [--candidates det|auto|lean|lean-try|llm] [--focus-line N] [--lean-oracle-per-node] [--lean-oracle-max-calls N] [--rollout-k N] [--dedup-goal-expansions] [--goal-first-k N] [--goal-meta-penalty N] [--depth-bonus N] [--fill-mode safe|strict|hybrid] [--max-candidates-per-node N] [--verify-k N] [--cache-dir <path> | --no-cache] [--profile] [--summary-level 0|1|2|3] [--report-md <path>] [--llm-summary] [--llm-summary-timeout-s N] [--llm-planner] [--llm-planner-timeout-s N] [--smt-precheck] [--smt-timeout-ms N] [--smt-seed N] [--goal-dump] [--llm-timeout-s N] [--escalate-llm] [--allow-sorry-candidates] [--include-trace] [--pick best|best-ok|best-progress] [--quiet] [--research-notes-file <path>] [--include-diff] [--diff-context N] [--output-diff <path>] [--write | --write-to <path>] [--include-raw-verify] [--output-json <path>]",
        "  suggest     --repo <path> --file <relpath> --lemma <name> [--timeout-s N] [--output-json <path>]",
        "  loop        --repo <path> --file <relpath> --lemma <name> [--max-iters N] [--timeout-s N] [--output-json <path>]",
        "  review-prompt --repo <path> [--scope auto|staged|worktree] [--max-total-bytes N] [--per-file-bytes N] [--transcript-bytes N] [--cache-version STR] [--cache-model STR] [--output-json <path>]",
        "  review-diff --repo <path> [--scope auto|staged|worktree] [--prompt-only] [--require-key] [--timeout-s N] [--no-verify|--verify-timeout-s N|--verify-max-files N] [--max-total-bytes N] [--per-file-bytes N] [--transcript-bytes N] [--cache-version STR] [--cache-model STR] [--output-json <path>]",
        "  llm-chat [--repo <path>] [--system <text> | --system-file <path>] [--user <text> | --user-file <path>] [--tools agent] [--max-tool-iters N] [--timeout-s N] [--require-key] [--output-json <path>]",
        "  lint-style  --repo <path> [--github] --module <Root> [--module <Root> ...]",
        "  report      --repo <path> --files <relpath>... [--timeout-s N] [--max-sorries N] [--context-lines N] [--include-raw-verify] [--output-html <path>]",
        "  research-ingest --input <path> [--output-json <path>]",
        "  research-attach --report-json <path> --research-notes <path> [--top-k N] [--output-json <path>]",
        "  context-pack --repo <path> --file <relpath> [--decl <name> | --line N] [--context-lines N] [--nearby-lines N] [--max-nearby N] [--max-imports N]",
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
    let rest = &args[2..];

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

    match cmd {
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
                    json!({"ok": true, "written": p.display().to_string()}).to_string()
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

            let payload = plc::build_rubberduck_prompt(
                &repo_root,
                &file,
                &lemma,
                diagnostics.as_deref(),
            )?;
            let out = serde_json::to_value(payload).map_err(|e| format!("json encode: {e}"))?;

            if let Some(p) = output_json {
                write_json(&p, &out)?;
                println!(
                    "{}",
                    json!({"ok": true, "written": p.display().to_string()}).to_string()
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
                println!("{}", json!({"ok": true, "written": p.display().to_string()}).to_string());
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
                println!("{}", json!({"ok": true, "written": p.display().to_string()}).to_string());
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
                    json!({"ok": true, "written": p.display().to_string()}).to_string()
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
                .ok_or_else(|| "missing --start-line".to_string())? as usize;
            let end_line = arg_u64(rest, "--end-line")
                .ok_or_else(|| "missing --end-line".to_string())? as usize;
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

            let patched = plc::patch_first_sorry_in_region(&original_text, start_line, end_line, &replacement)?;

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
                println!("{}", json!({"ok": true, "written": p.display().to_string()}).to_string());
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
                println!("{}", json!({"ok": true, "written": p.display().to_string()}).to_string());
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
                adapt_candidates_for_error, adapt_candidates_for_sorry_context, default_det_candidates,
                extract_initial_goal_block, hash_text, is_made_no_progress, parse_json_string_array,
                progress_score_key, sanitize_candidates, verify_score_key, hash_state_key,
            };

            let repo_root = arg_value(rest, "--repo")
                .ok_or_else(|| "missing --repo".to_string())
                .map(PathBuf::from)?;
            let file = arg_value(rest, "--file").ok_or_else(|| "missing --file".to_string())?;
            let timeout_s = arg_u64(rest, "--timeout-s").unwrap_or(120);
            let total_timeout_s = arg_u64(rest, "--total-timeout-s").unwrap_or(timeout_s).max(1);
            let log_level = arg_u64(rest, "--log-level").unwrap_or(1);
            let events_jsonl = arg_value(rest, "--events-jsonl").map(PathBuf::from);
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
            let candidates_mode = arg_value(rest, "--candidates").unwrap_or_else(|| "det".to_string());
            let lean_oracle_per_node = arg_flag(rest, "--lean-oracle-per-node");
            let lean_oracle_max_calls = arg_u64(rest, "--lean-oracle-max-calls").unwrap_or(12) as usize;
            let rollout_k = arg_u64(rest, "--rollout-k").unwrap_or(0) as usize;
            let dedup_goal_expansions = arg_flag(rest, "--dedup-goal-expansions");
            let goal_first_k = arg_u64(rest, "--goal-first-k").unwrap_or(0) as usize;
            let fill_mode_raw = arg_value(rest, "--fill-mode");
            let focus_line_override = arg_u64(rest, "--focus-line").map(|x| x as usize);
            let max_candidates_per_node = arg_u64(rest, "--max-candidates-per-node").map(|x| x as usize);
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
            let report_md = arg_value(rest, "--report-md").map(PathBuf::from);
            let llm_summary = arg_flag(rest, "--llm-summary");
            let llm_summary_timeout_s = arg_u64(rest, "--llm-summary-timeout-s").unwrap_or(20);
            let llm_planner = arg_flag(rest, "--llm-planner");
            // Only used when compiled with `--features planner`.
            #[cfg(feature = "planner")]
            let llm_planner_timeout_s = arg_u64(rest, "--llm-planner-timeout-s").unwrap_or(10);
            #[cfg(not(feature = "planner"))]
            let _llm_planner_timeout_s = arg_u64(rest, "--llm-planner-timeout-s").unwrap_or(10);
            let smt_precheck = arg_flag(rest, "--smt-precheck");
            // Only used when compiled with `--features smt`.
            #[cfg(feature = "smt")]
            let smt_timeout_ms = arg_u64(rest, "--smt-timeout-ms").unwrap_or(1500);
            #[cfg(not(feature = "smt"))]
            let smt_timeout_ms = {
                let _ = arg_u64(rest, "--smt-timeout-ms").unwrap_or(1500);
                0u64
            };
            #[cfg(feature = "smt")]
            let smt_seed = arg_u64(rest, "--smt-seed").unwrap_or(0);
            #[cfg(not(feature = "smt"))]
            let smt_seed = {
                let _ = arg_u64(rest, "--smt-seed").unwrap_or(0);
                0u64
            };
            let llm_timeout_s = arg_u64(rest, "--llm-timeout-s").unwrap_or(60);
            let goal_dump = arg_flag(rest, "--goal-dump");
            let escalate_llm = arg_flag(rest, "--escalate-llm");
            let allow_sorry_candidates = arg_flag(rest, "--allow-sorry-candidates");
            let include_trace = arg_flag(rest, "--include-trace");
            let pick = arg_value(rest, "--pick").unwrap_or_else(|| "best".to_string());
            let quiet = arg_flag(rest, "--quiet");
            let research_notes_file = arg_value(rest, "--research-notes-file").map(PathBuf::from);
            let include_diff = arg_flag(rest, "--include-diff");
            let diff_context = arg_u64(rest, "--diff-context").unwrap_or(3) as usize;
            let output_diff = arg_value(rest, "--output-diff").map(PathBuf::from);
            let write = arg_flag(rest, "--write");
            let write_to = arg_value(rest, "--write-to").map(PathBuf::from);
            let include_raw_verify = arg_flag(rest, "--include-raw-verify");
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);

            if write && write_to.is_some() {
                return Err("use only one of --write or --write-to".to_string());
            }
            #[cfg(not(feature = "smt"))]
            if smt_precheck {
                return Err("--smt-precheck requires building with cargo feature `smt`".to_string());
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

            let cache_dir = if no_cache {
                None
            } else if let Some(p) = cache_dir_opt {
                Some(if p.is_absolute() { p } else { repo_root.join(p) })
            } else {
                Some(repo_root.join(".generated").join("proofpatch-cache"))
            };

            let abs = repo_root.join(&file);
            if !abs.exists() {
                return Err(format!("File not found: {}", abs.display()));
            }
            let original_text = std::fs::read_to_string(&abs)
                .map_err(|e| format!("read {}: {e}", abs.display()))?;

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
            let mut events_by_kind: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
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
                    events_all.push(events_tail.last().cloned().unwrap_or(serde_json::Value::Null));
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
                    }
                }),
            );

            let mut goal_dump_v: Option<serde_json::Value> = None;
            let mut lean_suggest_v: Option<serde_json::Value> = None;
            if goal_dump || matches!(candidates_mode.trim(), "auto" | "lean") {
                if let Some(dur) = budget_dur(goal_dump_timeout_s) {
                    let gd = if let Some(fl) = focus_line_override {
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
            let goal_first_k = if candidates_mode == "lean-try" {
                // Default “goal-first” probing for LeanTree-ish behavior.
                if goal_first_k == 0 { 3 } else { goal_first_k }
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
                if arg_value(rest, "--lean-oracle-max-calls").is_some() { lean_oracle_max_calls } else { 24 }
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
                    "{indent}try (simp; done)\n{indent}try (aesop; done)\n{indent}try (omega; done)\n{indent}try (nlinarith; done)\n{indent}try (linarith; done)\n{indent}try (ring_nf; done)\n{indent}try (norm_num; done)\n{indent}sorry"
                )
            };

            // Single-line `first | ... | sorry` form for tactic holes (works well inside nested contexts).
            let safe_first_line = || -> String {
                "first | (simp; done) | (aesop; done) | (omega; done) | (nlinarith; done) | (linarith; done) | (ring_nf; done) | (norm_num; done) | sorry".to_string()
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
                    let mut derived = plc::derive_candidates_from_goal_pretty(pretty);
                    if derived.is_empty() {
                        derived = default_det_candidates();
                    }
                    sanitize_candidates(derived)
                } else {
                    sanitize_candidates(default_det_candidates())
                }
            } else if candidates_mode == "lean" {
                let ls = if let Some(dur) = budget_dur(oracle_timeout_s) {
                    rt.block_on(plc::lean_suggest_nearest(&repo_root, &file, dur)).ok()
                } else {
                    bailed_total_timeout = true;
                    record_event("bailout_total_timeout", json!({ "where": "lean_suggest_nearest" }));
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
                        xs = plc::derive_candidates_from_goal_pretty(pretty);
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
                    .map(|s| matches!(s.trim().to_lowercase().as_str(), "1" | "true" | "yes" | "y" | "on"))
                    .unwrap_or(false);

                let mut xs: Vec<String>;
                if seed_oracle {
                    record_event("oracle_seed_call", json!({ "timeout_s": oracle_timeout_s }));
                    let t0 = std::time::Instant::now();
                    let ls = if let Some(dur) = budget_dur(oracle_timeout_s) {
                        rt.block_on(plc::lean_suggest_nearest(&repo_root, &file, dur)).ok()
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
                    system.push_str("\n\nResearch notes (external; may be incomplete):\n");
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
                            let derived = plc::derive_candidates_from_goal_pretty(pretty);
                            if derived.is_empty() { None } else { Some(derived) }
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
                    "by\n  first | (simp; done) | (aesop; done) | (omega; done) | (nlinarith; done) | (linarith; done) | (ring_nf; done) | (norm_num; done) | sorry".to_string(),
                    "by\n  classical\n  first | (simp; done) | (aesop; done) | (omega; done) | (nlinarith; done) | (linarith; done) | (ring_nf; done) | (norm_num; done) | sorry".to_string(),
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
            let mut expanded_goal_hashes: std::collections::HashSet<u64> = std::collections::HashSet::new();

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
            let mut goal_dump_cache: std::collections::HashMap<(u64, usize, usize), (u64, usize, usize, String)> =
                std::collections::HashMap::new(); // (state_key, n_goals, hyps_total, target)
            // Companion cache for `hyps_texts` (used by SMT ranking and tactic reranking).
            let mut goal_dump_hyps_cache_hits: usize = 0;
            let mut goal_dump_hyps_cache_misses: usize = 0;
            let mut goal_dump_hyps_cache: std::collections::HashMap<(u64, usize, usize), Vec<String>> =
                std::collections::HashMap::new(); // (text_hash,len,line) -> hyps_texts
            // Interpretation of these counters:
            // - “hits” means we got `hyps_texts` without calling Lean (memory or disk).
            // - “misses” means we *wanted* `hyps_texts` (for SMT/ranking) but couldn't find them.
            // This is meant to track whether our “no extra Lean calls” design is actually working.

            #[cfg(feature = "planner")]
            let mut planner_cache: std::collections::HashMap<(u64, u64), plc::planner::PlannerDecision> =
                std::collections::HashMap::new(); // ((state_key, goal_sig) -> decision)
            #[cfg(feature = "planner")]
            let mut planner_cache_hits: u64 = 0;
            #[cfg(feature = "planner")]
            let mut planner_cache_misses: u64 = 0;
            #[cfg(feature = "planner")]
            let mut prof_planner_ms: u64 = 0;

            #[cfg(feature = "smt")]
            let mut smt_entails_cache: std::collections::HashMap<(u64, u64), bool> =
                std::collections::HashMap::new(); // ((state_key, goal_sig) -> entails)
            #[cfg(feature = "smt")]
            let mut smt_cache_hits: u64 = 0;
            #[cfg(feature = "smt")]
            let mut smt_cache_misses: u64 = 0;
            #[cfg(feature = "smt")]
            let mut prof_smt_ms: u64 = 0;
            #[cfg(not(feature = "smt"))]
            let smt_cache_hits: u64 = 0;
            #[cfg(not(feature = "smt"))]
            let smt_cache_misses: u64 = 0;
            #[cfg(not(feature = "smt"))]
            let _prof_smt_ms: u64 = 0;

            // Baseline verify (for first-error line; also returned in output).
            let (baseline_raw_v, baseline_summary, baseline_ms, baseline_skipped) = if let Some(dur) = budget_dur(timeout_s) {
                let t0 = std::time::Instant::now();
                let baseline = rt
                    .block_on(plc::verify_lean_file(&repo_root, &file, dur))
                    .map_err(|e| format!("verify failed: {e}"))?;
                let baseline_ms = t0.elapsed().as_millis() as u64;
                prof_verify_baseline_ms = prof_verify_baseline_ms.saturating_add(baseline_ms);
                prof_verify_baseline_calls += 1;
                let baseline_raw_v =
                    serde_json::to_value(baseline).map_err(|e| format!("serialize verify: {e}"))?;
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
                    stderr: "total timeout (budget exhausted before baseline verify)".to_string(),
                    cmd: vec![],
                    cwd: repo_root.display().to_string(),
                    tmp_file: None,
                };
                let raw_v = serde_json::to_value(raw).map_err(|e| format!("serialize verify: {e}"))?;
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
            let (focus_decl_name, focus_line_1) = {
                let locs0 = plc::locate_sorries_in_text(&original_text, 200, 1).unwrap_or_default();
                if let Some(fl) = focus_line_override {
                    let picked = locs0
                        .iter()
                        .min_by_key(|l| (l.line as i64 - fl as i64).abs())
                        .cloned();
                    (
                        picked.as_ref().and_then(|s| s.decl_name.clone()),
                        picked.map(|s| s.line),
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
                        picked.map(|s| s.line),
                    )
                }
            };
            record_event(
                "focus",
                json!({
                    "decl": focus_decl_name,
                    "line": focus_line_1,
                    "source": if focus_line_override.is_some() { "focus_line" } else { "first_error_or_primary_sorry" },
                }),
            );

            // Small helper for grokkable reporting.
            let classify_failure_mode = |first_error: Option<&str>| -> &'static str {
                let s = first_error.unwrap_or("").to_lowercase();
                if s.contains("unknown tactic") {
                    "unknown_tactic"
                } else if s.contains("synthinstancefailed") || s.contains("failed to synthesize instance") {
                    "typeclass_instance_failed"
                } else if s.contains("unsolved goals") {
                    "unsolved_goals"
                } else if s.contains("omega could not prove") || s.contains("no usable constraints") {
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
            let mut eval_cache: std::collections::HashMap<u64, CachedEval> = std::collections::HashMap::new();
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
                        record_event("bailout_total_timeout", json!({ "where": "frontier_prefill" }));
                        break;
                    }
                    // Disk cache (eval) prefill: if we have the full eval for this text, it supplies
                    // verify + sorry counts in one shot and avoids redundant work.
                    if (n.verify_summary.is_none() || n.sorries.is_none()) && n.verify_raw.is_some() == false {
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
                            prof_locate_sorries_ms =
                                prof_locate_sorries_ms.saturating_add(t0.elapsed().as_millis() as u64);
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
                                record_event("bailout_total_timeout", json!({ "where": "frontier_verify" }));
                                break;
                            };
                            let t0 = std::time::Instant::now();
                            let raw = rt
                                .block_on(plc::verify_lean_text(
                                    &repo_root,
                                    &n.text,
                                    dur,
                                ))
                                .map_err(|e| format!("verify failed: {e}"))?;
                            prof_verify_nodes_ms =
                                prof_verify_nodes_ms.saturating_add(t0.elapsed().as_millis() as u64);
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
                    let mut ka = verify_score_key(sa, a.sorries.unwrap_or(999), a.conservative_sorries.unwrap_or(999));
                    let mut kb = verify_score_key(sb, b.sorries.unwrap_or(999), b.conservative_sorries.unwrap_or(999));
                    if depth_bonus > 0 {
                        // Lower is better; subtracting rewards deeper nodes slightly (tie-break/escape hatch).
                        ka.1 = ka.1.saturating_sub(depth_bonus.saturating_mul(a.depth as i64));
                        kb.1 = kb.1.saturating_sub(depth_bonus.saturating_mul(b.depth as i64));
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
                    let locs_all = plc::locate_sorries_in_text(&parent.text, 200, 1).unwrap_or_default();
                    let locs = if let Some(dn) = parent.focus_decl_name.as_deref() {
                        let xs: Vec<plc::SorryLocation> = locs_all
                            .iter()
                            .cloned()
                            .filter(|l| l.decl_name.as_deref() == Some(dn))
                            .collect();
                        if xs.is_empty() { locs_all } else { xs }
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
                            let mut goal_sig_opt = goal_dump_cache
                                .get(&k)
                                .and_then(|(_, _, _, tgt)| if tgt.is_empty() { None } else { Some(hash_text(tgt)) });
                            // Disk fallback (still cache-only).
                            if goal_sig_opt.is_none() {
                                if let Some(cd) = cache_dir.as_ref() {
                                    if let Some(tup) = cache_read_goal_dump(cd, th, parent.text.len(), l.line) {
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
                        if xs.is_empty() { locs } else { xs }
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
                    let selected = if candidates_mode == "lean-try" && goal_first_k > 0 && !locs.is_empty() {
                        // Goal-first scheduler (bounded): probe a few candidate holes with a cheap goal dump,
                        // compute a state key and a crude "difficulty" score, then pick the easiest.
                        let fl = parent.focus_line.or(first_error_line_1).unwrap_or(locs[0].line);
                        let mut cands: Vec<plc::SorryLocation> = locs.clone();
                        cands.sort_by_key(|l| (l.line as i64 - fl as i64).abs());
                        cands.truncate(goal_first_k);

                        // Optional LLM planner: can override which hole to focus and how much oracle budget to spend.
                        // This is feature-gated and defaults off.
                        let planner_selected: Option<plc::SorryLocation> = if llm_planner {
                            #[cfg(not(feature = "planner"))]
                            {
                                return Err("--llm-planner requires building with cargo feature `planner`".to_string());
                            }
                            #[cfg(feature = "planner")]
                            {
                                let mut picked_sel: Option<plc::SorryLocation> = None;

                                // Probe only the closest hole to build planner evidence.
                                let seed = cands.first().cloned().unwrap_or_else(|| locs[0].clone());
                                let th = hash_text(&parent.text);
                                let key = (th, parent.text.len(), seed.line);

                                // Ensure we have (state_key,n_goals,hyps_total,target) for the seed.
                                let mut target: String = String::new();
                                let (state_key, n_goals, hyps_total, target_cached) = if let Some(v) = goal_dump_cache.get(&key) {
                                    v.clone()
                                } else {
                                    if lean_oracle_calls >= lean_oracle_max_calls {
                                        // No budget to probe; skip planner.
                                        (UNKNOWN_STATE_KEY, 0usize, 0usize, String::new())
                                    } else {
                                        goal_dump_calls += 1;
                                        lean_oracle_calls += 1;
                                        let t0 = std::time::Instant::now();
                                        let gd = if let Some(dur) = budget_dur(goal_dump_timeout_s) {
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
                                        prof_goal_dump_ms = prof_goal_dump_ms.saturating_add(elapsed_ms);

                                        let pp = gd.as_ref().and_then(|v| v.get("pp_dump"));
                                        let state_key = pp.and_then(|pp| hash_state_key(pp)).unwrap_or(UNKNOWN_STATE_KEY);
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
                                                    .map(|g| g.get("hyps").and_then(|h| h.as_array()).map(|x| x.len()).unwrap_or(0))
                                                    .sum::<usize>()
                                            })
                                            .unwrap_or(0);
                                        target = pp
                                            .and_then(|pp| pp.get("goals"))
                                            .and_then(|v| v.as_array())
                                            .and_then(|a| a.first())
                                            .and_then(|g| g.get("pretty"))
                                            .and_then(|v| v.as_str())
                                            .and_then(|s| s.lines().find_map(|ln| ln.trim_start().strip_prefix("⊢").map(|r| r.trim().to_string())))
                                            .unwrap_or_default();
                                        let hyps_texts: Vec<String> = pp
                                            .and_then(|pp| pp.get("goals"))
                                            .and_then(|v| v.as_array())
                                            .and_then(|a| a.first())
                                            .and_then(|g| g.get("hyps"))
                                            .and_then(|v| v.as_array())
                                            .map(|a| {
                                                a.iter()
                                                    .filter_map(|h| h.get("text").and_then(|v| v.as_str()).map(|s| s.to_string()))
                                                    .collect::<Vec<_>>()
                                            })
                                            .unwrap_or_default();
                                        let tup = (state_key, n_goals, hyps_total, target.clone());
                                        goal_dump_cache.insert(key, tup.clone());
                                        goal_dump_hyps_cache.insert(key, hyps_texts.clone());
                                        if let Some(cd) = cache_dir.as_ref() {
                                            cache_write_goal_dump(cd, th, parent.text.len(), seed.line, state_key, n_goals, hyps_total, &target, &hyps_texts);
                                        }
                                        tup
                                    }
                                };

                                if target.is_empty() {
                                    target = target_cached;
                                }
                                let goal_sig = hash_text(&target);
                                let cache_key = (state_key, goal_sig);

                                let decision = if let Some(d) = planner_cache.get(&cache_key).cloned() {
                                    planner_cache_hits += 1;
                                    Some(d)
                                } else if let Some(cd) = cache_dir.as_ref() {
                                    if let Some(v) = cache_read_planner(cd, state_key, goal_sig) {
                                        if let Ok(d) = serde_json::from_value::<plc::planner::PlannerDecision>(v.clone()) {
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
                                    let ev_json = serde_json::to_string(&evidence).unwrap_or_else(|_| "{}".to_string());
                                    let t0 = std::time::Instant::now();
                                    let res = rt.block_on(plc::planner::plan(
                                        system,
                                        &ev_json,
                                        StdDuration::from_secs(llm_planner_timeout_s),
                                    ));
                                    prof_planner_ms = prof_planner_ms.saturating_add(t0.elapsed().as_millis() as u64);
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
                                        std::env::set_var("PROOFPATCH_ORACLE_PASSES", passes.to_string());
                                    }
                                    if !decision.oracle_tactics.is_empty() {
                                        std::env::set_var("PROOFPATCH_ORACLE_TACTICS", decision.oracle_tactics.join(","));
                                    }
                                    if !decision.ban_oracle_tactics.is_empty() {
                                        std::env::set_var("PROOFPATCH_ORACLE_BAN", decision.ban_oracle_tactics.join(","));
                                    }
                                    if let Some(fl1) = decision.focus_line_1 {
                                        if let Some(picked) = cands.iter().find(|h| h.line as u64 == fl1).cloned() {
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
                            let (state_key, n_goals, hyps_total, _target) = if let Some(v) = goal_dump_cache.get(&key) {
                                goal_dump_cache_hits += 1;
                                v.clone()
                            } else if let Some(cd) = cache_dir.as_ref() {
                                if let Some(tup) = cache_read_goal_dump(cd, th, parent.text.len(), s0.line) {
                                    goal_dump_cache_hits += 1;
                                    goal_dump_cache.insert(key, tup.clone());
                                    if let Some(hyps) = cache_read_goal_dump_hyps_texts(cd, th, parent.text.len(), s0.line) {
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
                                            json!({ "where": "goal_dump_goal_first" }),
                                        );
                                        None
                                    };
                                    let elapsed_ms = t0.elapsed().as_millis() as u64;
                                    last_probe_ms = Some(elapsed_ms);
                                    prof_goal_dump_ms = prof_goal_dump_ms.saturating_add(elapsed_ms);
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
                                                .map(|g| g.get("hyps").and_then(|h| h.as_array()).map(|x| x.len()).unwrap_or(0))
                                                .sum::<usize>()
                                        })
                                        .unwrap_or(0);
                                    let target = pp
                                        .and_then(|pp| pp.get("goals"))
                                        .and_then(|v| v.as_array())
                                        .and_then(|a| a.first())
                                        .and_then(|g| g.get("pretty"))
                                        .and_then(|v| v.as_str())
                                        .and_then(|s| s.lines().find_map(|ln| ln.trim_start().strip_prefix("⊢").map(|r| r.trim().to_string())))
                                        .unwrap_or_default();
                                    let hyps_texts: Vec<String> = pp
                                        .and_then(|pp| pp.get("goals"))
                                        .and_then(|v| v.as_array())
                                        .and_then(|a| a.first())
                                        .and_then(|g| g.get("hyps"))
                                        .and_then(|v| v.as_array())
                                        .map(|a| {
                                            a.iter()
                                                .filter_map(|h| h.get("text").and_then(|v| v.as_str()).map(|s| s.to_string()))
                                                .collect::<Vec<_>>()
                                        })
                                        .unwrap_or_default();
                                    let tup = (state_key, n_goals, hyps_total, target.clone());
                                    goal_dump_cache.insert(key, tup.clone());
                                    goal_dump_hyps_cache.insert(key, hyps_texts.clone());
                                    cache_write_goal_dump(cd, th, parent.text.len(), s0.line, state_key, n_goals, hyps_total, &target, &hyps_texts);

                                    // SMT precheck: if hyps entail target (LIA-only), take this hole immediately.
                                    #[cfg(feature = "smt")]
                                    if smt_precheck {
                                        if let Some(pp0) = pp {
                                            let goal_sig = hash_text(&target);
                                            let ck = (state_key, goal_sig);
                                            let mut entails_opt: Option<bool> = None;
                                            if let Some(v) = smt_entails_cache.get(&ck).copied() {
                                                smt_cache_hits += 1;
                                                entails_opt = Some(v);
                                            } else if let Some(cd2) = cache_dir.as_ref() {
                                                if let Some(v) = cache_read_smt_entails(cd2, state_key, goal_sig) {
                                                    smt_cache_hits += 1;
                                                    smt_entails_cache.insert(ck, v);
                                                    entails_opt = Some(v);
                                                }
                                            }
                                            if entails_opt.is_none() {
                                                smt_cache_misses += 1;
                                                let t_smt0 = std::time::Instant::now();
                                                let entails = smt_entails_from_pp_dump(pp0, smt_timeout_ms, smt_seed).unwrap_or(None);
                                                prof_smt_ms = prof_smt_ms.saturating_add(t_smt0.elapsed().as_millis() as u64);
                                                if let Some(b) = entails {
                                                    smt_entails_cache.insert(ck, b);
                                                    if let Some(cd2) = cache_dir.as_ref() {
                                                        cache_write_smt_entails(cd2, state_key, goal_sig, b);
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
                                prof_goal_dump_ms = prof_goal_dump_ms.saturating_add(elapsed_ms);
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
                                            .map(|g| g.get("hyps").and_then(|h| h.as_array()).map(|x| x.len()).unwrap_or(0))
                                            .sum::<usize>()
                                    })
                                    .unwrap_or(0);
                                let target = pp
                                    .and_then(|pp| pp.get("goals"))
                                    .and_then(|v| v.as_array())
                                    .and_then(|a| a.first())
                                    .and_then(|g| g.get("pretty"))
                                    .and_then(|v| v.as_str())
                                    .and_then(|s| s.lines().find_map(|ln| ln.trim_start().strip_prefix("⊢").map(|r| r.trim().to_string())))
                                    .unwrap_or_default();
                                let hyps_texts: Vec<String> = pp
                                    .and_then(|pp| pp.get("goals"))
                                    .and_then(|v| v.as_array())
                                    .and_then(|a| a.first())
                                    .and_then(|g| g.get("hyps"))
                                    .and_then(|v| v.as_array())
                                    .map(|a| {
                                        a.iter()
                                            .filter_map(|h| h.get("text").and_then(|v| v.as_str()).map(|s| s.to_string()))
                                            .collect::<Vec<_>>()
                                    })
                                    .unwrap_or_default();
                                let tup = (state_key, n_goals, hyps_total, target.clone());
                                goal_dump_cache.insert(key, tup.clone());
                                goal_dump_hyps_cache.insert(key, hyps_texts.clone());
                                if let Some(cd) = cache_dir.as_ref() {
                                    cache_write_goal_dump(cd, th, parent.text.len(), s0.line, state_key, n_goals, hyps_total, &target, &hyps_texts);
                                }

                                // SMT precheck (same logic as above, but after the no-cache path).
                                #[cfg(feature = "smt")]
                                if smt_precheck {
                                    if let Some(pp0) = pp {
                                        let goal_sig = hash_text(&target);
                                        let ck = (state_key, goal_sig);
                                        let mut entails_opt: Option<bool> = None;
                                        if let Some(v) = smt_entails_cache.get(&ck).copied() {
                                            smt_cache_hits += 1;
                                            entails_opt = Some(v);
                                        } else if let Some(cd2) = cache_dir.as_ref() {
                                            if let Some(v) = cache_read_smt_entails(cd2, state_key, goal_sig) {
                                                smt_cache_hits += 1;
                                                smt_entails_cache.insert(ck, v);
                                                entails_opt = Some(v);
                                            }
                                        }
                                        if entails_opt.is_none() {
                                            smt_cache_misses += 1;
                                            let t_smt0 = std::time::Instant::now();
                                            let entails = smt_entails_from_pp_dump(pp0, smt_timeout_ms, smt_seed).unwrap_or(None);
                                            prof_smt_ms = prof_smt_ms.saturating_add(t_smt0.elapsed().as_millis() as u64);
                                            if let Some(b) = entails {
                                                smt_entails_cache.insert(ck, b);
                                                if let Some(cd2) = cache_dir.as_ref() {
                                                    cache_write_smt_entails(cd2, state_key, goal_sig, b);
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
                            let unknown_pen = if state_key == UNKNOWN_STATE_KEY { 10_000 } else { 0 };
                            let meta_vars_pen = if goal_meta_penalty > 0 {
                                // If the target contains metavariables, treat it as harder to avoid wasting budget.
                                let th = hash_text(&parent.text);
                                let k = (th, parent.text.len(), s0.line);
                                let target = goal_dump_cache
                                    .get(&k)
                                    .map(|(_, _, _, t)| t.clone())
                                    .unwrap_or_default();
                                let meta = target.matches("?m").count() + target.matches("?_").count();
                                (meta as i64).saturating_mul(goal_meta_penalty)
                            } else {
                                0
                            };
                            let score0 =
                                unknown_pen as i64 + (n_goals as i64 * 100) + hyps_total as i64 + meta_vars_pen;

                            // Optional SMT hint: if the target is implied by (some) integer constraints in the local context,
                            // treat this hole as very easy (it likely succumbs to `omega`/`linarith`/`simp`).
                            let score = {
                                #[cfg(feature = "smt")]
                                {
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
                                            let ck = (state_key, goal_sig);
                                            let mut cached = smt_entails_cache.get(&ck).copied();
                                            if cached.is_none() {
                                                if let Some(cd) = cache_dir.as_ref() {
                                                    if let Some(ent) = cache_read_smt_entails(cd, state_key, goal_sig) {
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
                                                let hyps_texts = if let Some(xs) = goal_dump_hyps_cache.get(&k).cloned() {
                                                    goal_dump_hyps_cache_hits += 1;
                                                    xs
                                                } else if let Some(cd) = cache_dir.as_ref() {
                                                    if let Some(xs) =
                                                        cache_read_goal_dump_hyps_texts(cd, th, parent.text.len(), s0.line)
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
                                                    let ent = smt_entails_from_hyps_target(
                                                        &hyps_texts,
                                                        &target,
                                                        smt_timeout_ms,
                                                        smt_seed,
                                                    )
                                                    .unwrap_or(None);
                                                    prof_smt_ms =
                                                        prof_smt_ms.saturating_add(t0.elapsed().as_millis() as u64);
                                                    if let Some(ent) = ent {
                                                        smt_entails_cache.insert(ck, ent);
                                                        if let Some(cd) = cache_dir.as_ref() {
                                                            cache_write_smt_entails(cd, state_key, goal_sig, ent);
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
                                }
                                #[cfg(not(feature = "smt"))]
                                {
                                    score0
                                }
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
                                let ind_k = l.chars().take_while(|c| *c == ' ' || *c == '\t').count();
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
                    if candidates_mode == "lean-try" && lean_oracle_per_node && lean_oracle_calls < lean_oracle_max_calls {
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
                                record_event("bailout_total_timeout", json!({ "where": "oracle_call" }));
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
                            let verify_ok = verify_raw.and_then(|v| v.get("ok")).and_then(|v| v.as_bool());
                            let verify_timeout =
                                verify_raw.and_then(|v| v.get("timeout")).and_then(|v| v.as_bool());
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
                            record_event(
                                "oracle_result",
                                json!({
                                    "ms": oracle_ms,
                                    "ok": ls.is_some(),
                                    "error": ls_err,
                                    "suggestions_n": suggs.len(),
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
                                    .and_then(|s| s.lines().find_map(|ln| ln.trim_start().strip_prefix("⊢").map(|r| r.trim().to_string())))
                                    .unwrap_or_default();
                                let sk = goal_hash.unwrap_or(UNKNOWN_STATE_KEY);
                                goal_dump_cache.insert(key, (sk, n_goals, hyps_total, target.clone()));
                                // Best-effort: record local-context lines so SMT precheck can run without re-calling Lean.
                                let hyps_texts: Vec<String> = pp
                                    .get("goals")
                                    .and_then(|v| v.as_array())
                                    .and_then(|a| a.first())
                                    .and_then(|g| g.get("hyps"))
                                    .and_then(|v| v.as_array())
                                    .map(|a| {
                                        a.iter()
                                            .filter_map(|h| h.get("text").and_then(|v| v.as_str()).map(|s| s.to_string()))
                                            .collect::<Vec<_>>()
                                    })
                                    .unwrap_or_default();
                                goal_dump_hyps_cache.insert(key, hyps_texts.clone());
                                if let Some(cd) = cache_dir.as_ref() {
                                    cache_write_goal_dump(cd, h, parent.text.len(), focus_line, sk, n_goals, hyps_total, &target, &hyps_texts);
                                }
                                // Use first goal pretty as a cheap heuristic source.
                                if let Some(pretty) = pp
                                    .get("goals")
                                    .and_then(|v| v.as_array())
                                    .and_then(|xs| xs.first())
                                    .and_then(|v| v.get("pretty"))
                                    .and_then(|v| v.as_str())
                                {
                                    derived = plc::derive_candidates_from_goal_pretty(pretty);
                                }
                                // Also derive from the target + hypothesis snippets (often includes type hints).
                                if !target.trim().is_empty() {
                                    let mut more = plc::derive_candidates_from_goal_context(&hyps_texts, &target);
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
                                    lean_state_candidates_cache.entry(sk).or_insert_with(|| xs.clone());
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
                    let candidates_here0 = adapt_candidates_for_error(base_candidates, parent_first_error);
                    let candidates_here0 =
                        adapt_candidates_for_sorry_context(&candidates_here0, &sel.line_text, is_tactic_context);

                    // Optional: if deterministic tactics stalled, opportunistically ask the LLM
                    // for more candidates for this exact region.
                    let candidates_here = if escalate_llm
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
                            system.push_str("\n\nResearch notes (external; may be incomplete):\n");
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
                                llm_escalate_last_error = Some("llm_response_not_json_string_array".to_string());
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
                    #[cfg(feature = "smt")]
                    let (smt_entails_opt, smt_hint_json): (Option<bool>, Option<serde_json::Value>) = if smt_precheck {
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
                                let ck = (sk, goal_sig);
                                if let Some(v) = smt_entails_cache.get(&ck).copied() {
                                    smt_cache_hits += 1;
                                    (
                                        Some(v),
                                        Some(json!({
                                            "entails": v,
                                            "source": "mem",
                                            "state_key": sk,
                                            "goal_sig": goal_sig,
                                        })),
                                    )
                                } else if let Some(v) = cache_read_smt_entails(cd, sk, goal_sig) {
                                    smt_cache_hits += 1;
                                    smt_entails_cache.insert(ck, v);
                                    (
                                        Some(v),
                                        Some(json!({
                                            "entails": v,
                                            "source": "disk",
                                            "state_key": sk,
                                            "goal_sig": goal_sig,
                                        })),
                                    )
                                } else {
                                    // Try to compute from cached hyps_texts + target (no Lean).
                                    smt_cache_misses += 1;
                                    let hyps_texts = if let Some(xs) = goal_dump_hyps_cache.get(&k).cloned() {
                                        goal_dump_hyps_cache_hits += 1;
                                        xs
                                    } else if let Some(xs) =
                                        cache_read_goal_dump_hyps_texts(cd, th, parent.text.len(), sel.line)
                                    {
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
                                        let ent = smt_entails_from_hyps_target(
                                            &hyps_texts,
                                            &target,
                                            smt_timeout_ms,
                                            smt_seed,
                                        )
                                        .unwrap_or(None);
                                        prof_smt_ms = prof_smt_ms.saturating_add(t0.elapsed().as_millis() as u64);
                                        if let Some(ent) = ent {
                                            smt_entails_cache.insert(ck, ent);
                                            cache_write_smt_entails(cd, sk, goal_sig, ent);
                                            (
                                                Some(ent),
                                                Some(json!({
                                                    "entails": ent,
                                                    "source": "computed",
                                                    "state_key": sk,
                                                    "goal_sig": goal_sig,
                                                })),
                                            )
                                        } else {
                                            (
                                                None,
                                                Some(json!({
                                                    "entails": serde_json::Value::Null,
                                                    "source": "computed_unknown",
                                                    "state_key": sk,
                                                    "goal_sig": goal_sig,
                                                })),
                                            )
                                        }
                                    }
                                }
                            } else {
                                (
                                    None,
                                    Some(json!({
                                        "entails": serde_json::Value::Null,
                                        "source": "no_cache_dir",
                                        "state_key": sk,
                                        "goal_sig": serde_json::Value::Null,
                                    })),
                                )
                            }
                        } else {
                            (None, None)
                        }
                    } else {
                        (None, None)
                    };
                    #[cfg(not(feature = "smt"))]
                    let smt_entails_opt: Option<bool> = None;
                    #[cfg(not(feature = "smt"))]
                    let smt_hint_json: Option<serde_json::Value> = None;

                    // If the goal looks “wide” or “context heavy”, we should prefer low-branching
                    // candidates even more aggressively.
                    let (n_goals, hyps_total, meta_vars_target) = {
                        let th = hash_text(&parent.text);
                        let k = (th, parent.text.len(), sel.line);
                        goal_dump_cache
                            .get(&k)
                            .map(|(_, ng, ht, target)| {
                                let meta = target.matches("?m").count() + target.matches("?_").count();
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
                            let smt_bonus: i64 = match smt_entails_opt {
                                Some(true) if is_arith => -5_000,
                                Some(false) if is_arith => 5_000,
                                _ => 0,
                            };
                            // Note: `smt_bonus` is deliberately coarse. We only want SMT to act as a
                            // *tie-breaker* or a strong nudge, not to dominate the learned/state-action prior.
                            let lines = c.lines().count() as i64;
                            let holes = c.matches("?_").count() as i64;
                            let bullets = c.matches("\n·").count() as i64;
                            let len = c.chars().count() as i64;
                            let complexity =
                                holes * (50 + 10 * n_goals) + bullets * 10 + lines * 5 + len / 40 + hyps_total / 20;
                            CandRank {
                                cand: c,
                                cand_h,
                                prior,
                                complexity,
                                smt_bonus,
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
                        .map(|s| matches!(s.trim().to_lowercase().as_str(), "0" | "false" | "no" | "n" | "off"))
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
                                match smt_entails_opt {
                                    Some(true) => "smt_entails_true_boost",
                                    Some(false) => "smt_entails_false_penalize",
                                    None => "no_smt_signal",
                                }
                            } else if r.is_arith {
                                match smt_entails_opt {
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
                                "second_key": r.complexity + r.smt_bonus,
                                "reason": smt_reason,
                            }));
                        }
                        Some(json!({
                            "line": sel.line,
                            "n_goals": n_goals,
                            "hyps_total": hyps_total,
                            "meta_vars_target": meta_vars_target,
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
                            let locs_r = plc::locate_sorries_in_text(&rolled_text, 200, 1).unwrap_or_default();
                            prof_locate_sorries_ms =
                                prof_locate_sorries_ms.saturating_add(t0.elapsed().as_millis() as u64);
                            if locs_r.is_empty() {
                                break;
                            }
                            let next = locs_r
                                .iter()
                                .min_by_key(|l| (l.line as i64 - rolled_line as i64).abs())
                                .cloned();
                            let Some(sel_r) = next else { break; };

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
                            let Ok(patched_r) = patched_r else { break; };
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
                                        record_event("bailout_total_timeout", json!({ "where": "verify_node" }));
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
                                record_event("bailout_total_timeout", json!({ "where": "verify_node" }));
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
                            let locs2 =
                                plc::locate_sorries_in_text(&rolled_text, 500, 1).unwrap_or_default();
                            prof_locate_sorries_ms =
                                prof_locate_sorries_ms.saturating_add(t0.elapsed().as_millis() as u64);
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
                                .and_then(|(_, _, _, tgt)| if tgt.is_empty() { None } else { Some(hash_text(tgt)) })
                                .or(parent.focus_goal_sig)
                        };
                        new_frontier.push(Node {
                            id: next_id,
                            depth: parent.depth + 1,
                            text: rolled_text.clone(),
                            focus_decl_name: parent.focus_decl_name.clone().or_else(|| sel.decl_name.clone()),
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
            let best = if let Some(d) = best_done {
                d
            } else if all.is_empty() {
                // If we bailed early (e.g. total-timeout) before moving any nodes into `all`,
                // fall back to the root node so we can still emit a useful partial result.
                frontier
                    .first()
                    .cloned()
                    .unwrap_or_else(|| Node {
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
                        sorries: Some(plc::locate_sorries_in_text(&original_text, 500, 1).unwrap_or_default().len()),
                        conservative_sorries: Some(plc::count_sorry_tokens_conservative(&original_text).unwrap_or(0)),
                        smt_hint: None,
                        rank_hint: None,
                    })
            } else {
                let mut xs = all.clone();
                xs.sort_by(|a, b| {
                    let sa = a.verify_summary.as_ref().unwrap();
                    let sb = b.verify_summary.as_ref().unwrap();
                    let ka = verify_score_key(sa, a.sorries.unwrap_or(999), a.conservative_sorries.unwrap_or(999));
                    let kb = verify_score_key(sb, b.sorries.unwrap_or(999), b.conservative_sorries.unwrap_or(999));
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
                    let ka = progress_score_key(sa, a.sorries.unwrap_or(999), a.conservative_sorries.unwrap_or(999));
                    let kb = progress_score_key(sb, b.sorries.unwrap_or(999), b.conservative_sorries.unwrap_or(999));
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

            // Finalize event stream before rendering any human summaries.
            // We also drop the recorder closure so we can immutably read `events_tail` safely.
            record_event(
                "end",
                json!({
                    "bailouts": { "total_timeout": bailed_total_timeout },
                    "remaining_ms": remaining_ms(run_deadline),
                }),
            );
            drop(record_event);

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
                    eprintln!("[tree-search-nearest] focus decl={dn} line={}", focus_line_1.unwrap_or(0));
                }
                if bailed_total_timeout {
                    eprintln!("[tree-search-nearest] bailout total_timeout=true remaining_ms={}", remaining_ms(run_deadline));
                }

                if log_level >= 2 {
                    let oracle_n = events_by_kind.get("oracle_call").copied().unwrap_or(0);
                    let verify_node_n = events_by_kind.get("verify_node").copied().unwrap_or(0);
                    let bailout_n = events_by_kind.get("bailout_total_timeout").copied().unwrap_or(0);
                    let seed_call_n = events_by_kind.get("oracle_seed_call").copied().unwrap_or(0);
                    let seed_result_n = events_by_kind.get("oracle_seed_result").copied().unwrap_or(0);
                    let seed_skipped_n = events_by_kind.get("oracle_seed_skipped").copied().unwrap_or(0);
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
                        let include_seed_skipped = bailed_total_timeout || seed_call_n > 0 || seed_result_n > 0;
                        let is_noise_kind = |k: &str| -> bool {
                            k == "start" || k == "budgets" || (!include_seed_skipped && k == "oracle_seed_skipped")
                        };
                        let filtered: Vec<&serde_json::Value> = events_tail
                            .iter()
                            .filter(|ev| {
                                let k = ev.get("kind").and_then(|v| v.as_str()).unwrap_or("");
                                !is_noise_kind(k)
                            })
                            .collect();
                        let list = if filtered.is_empty() { events_tail.iter().collect() } else { filtered };

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
                                let ok = ev.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
                                let n = ev.get("suggestions_n").and_then(|v| v.as_u64()).unwrap_or(0);
                                let ms = ev.get("ms").and_then(|v| v.as_u64()).unwrap_or(0);
                                format!("ok={ok} n={n} ms={ms}")
                            }
                            "verify_node" => {
                                let ok = ev.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
                                let s = ev.get("sorries").and_then(|v| v.as_u64()).unwrap_or(0);
                                let cache = ev.get("cache").and_then(|v| v.as_str()).unwrap_or("?");
                                let ms_opt = ev.get("ms").and_then(|v| v.as_u64());
                                if let Some(ms) = ms_opt {
                                    format!("ok={ok} sorries={s} ms={ms} cache={cache}")
                                } else {
                                    format!("ok={ok} sorries={s} cache={cache}")
                                }
                            }
                            "end" => {
                                let rem = ev.get("remaining_ms").and_then(|v| v.as_u64()).unwrap_or(0);
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
                        let ka = verify_score_key(sa, a.sorries.unwrap_or(999), a.conservative_sorries.unwrap_or(999));
                        let kb = verify_score_key(sb, b.sorries.unwrap_or(999), b.conservative_sorries.unwrap_or(999));
                        ka.cmp(&kb).then_with(|| a.id.cmp(&b.id))
                    });
                    eprintln!("[tree-search-nearest] trace nodes={} (showing top 6)", all.len());
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
                let target = write_to.as_ref().unwrap_or(&abs);
                if let Some(parent) = target.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| format!("failed to create dir {}: {}", parent.display(), e))?;
                }
                std::fs::write(target, picked.text.as_bytes())
                    .map_err(|e| format!("write {}: {e}", target.display()))?;
                written_file = Some(target.display().to_string());
            }

            let mut diff_written: Option<String> = None;
            let mut diff_unified: serde_json::Value = serde_json::Value::Null;
            if include_diff || output_diff.is_some() {
                let (d, truncated) = unified_diff_bounded(&original_text, &picked.text, diff_context, 120_000);
                if let Some(p) = output_diff.as_ref() {
                    if let Some(parent) = p.parent() {
                        std::fs::create_dir_all(parent)
                            .map_err(|e| format!("failed to create dir {}: {}", parent.display(), e))?;
                    }
                    std::fs::write(p, d.as_bytes()).map_err(|e| format!("write diff {}: {e}", p.display()))?;
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
            if let Some(p) = report_md.as_ref() {
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
                md.push_str(&format!("- picked ok/errors/sw/sorries: `{}/{}/{}/{}`\n",
                    ps.get("ok").and_then(|v| v.as_bool()).unwrap_or(false),
                    ps.get("counts").and_then(|c| c.get("errors")).and_then(|v| v.as_u64()).unwrap_or(0),
                    ps.get("counts").and_then(|c| c.get("sorry_warnings")).and_then(|v| v.as_u64()).unwrap_or(0),
                    picked.sorries.unwrap_or(999),
                ));
                md.push_str(&format!("- baseline sorries: `{}`\n", baseline_sorries));
                if let Some(dn) = focus_decl_name.as_deref() {
                    md.push_str(&format!("- focus decl: `{}` (line {})\n", dn, focus_line_1.unwrap_or(0)));
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
                            let okv = n.get("verify").and_then(|v| v.get("summary")).and_then(|v| v.get("ok")).and_then(|v| v.as_bool()).unwrap_or(false);
                            let errs = n.get("verify").and_then(|v| v.get("summary")).and_then(|v| v.get("counts")).and_then(|v| v.get("errors")).and_then(|v| v.as_u64()).unwrap_or(0);
                            let sw = n.get("verify").and_then(|v| v.get("summary")).and_then(|v| v.get("counts")).and_then(|v| v.get("sorry_warnings")).and_then(|v| v.as_u64()).unwrap_or(0);
                            let repl = n.get("last_replacement").and_then(|v| v.as_str()).unwrap_or("").replace('\n', " ");
                            md.push_str(&format!("- id={} depth={} ok={} e={} sw={} sorries={} repl=`{}`\n", id, depth, okv, errs, sw, sorries, repl));
                        }
                    }
                }
                md.push_str("\n### Next actions\n\n");
                md.push_str("- Re-run with `--summary-level 3 --include-trace` for a compact top-nodes view.\n");
                md.push_str("- Re-run with `--report-md <path>` to persist this report alongside JSON.\n");
                md.push_str("- If using `lean-try`, consider increasing `--lean-oracle-max-calls` and `--max-nodes`.\n");
                md.push_str("- If the search is bouncing between holes, increase `--goal-first-k` (default 3 for lean-try).\n");
                md.push_str("- If you want to prioritize actually closing goals, try `--fill-mode strict` or `--fill-mode hybrid`.\n");
                md.push_str("- If you want exploration, enable `--escalate-llm` (and configure provider).\n");
                if let Some(parent) = p.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| format!("failed to create dir {}: {}", parent.display(), e))?;
                }
                std::fs::write(p, md.as_bytes()).map_err(|e| format!("write report {}: {e}", p.display()))?;
            }

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
                "write_mode": if write { "inplace" } else if write_to.is_some() { "to_path" } else { "none" },
                "diff": diff_unified,
                "diff_written": diff_written,
                "artifacts": {
                    "events_jsonl": events_jsonl_written,
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
                    "smt_timeout_ms": smt_timeout_ms,
                    "smt_seed": smt_seed,
                    "escalate_llm": escalate_llm,
                    "allow_sorry_candidates": allow_sorry_candidates,
                    "include_trace": include_trace,
                    "pick": pick,
                    "research_notes": research_notes,
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
                        "goal_dump_hyps_cache_hits": goal_dump_hyps_cache_hits,
                        "goal_dump_hyps_cache_misses": goal_dump_hyps_cache_misses
                    }
                },
                "baseline_verify": { "summary": baseline_summary },
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
                    "sorries": best.sorries,
                    "conservative_sorries": best.conservative_sorries,
                    "focus_goal_sig": best.focus_goal_sig,
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
                    "sorries": best_progress.sorries,
                    "conservative_sorries": best_progress.conservative_sorries,
                    "focus_goal_sig": best_progress.focus_goal_sig,
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
                    "sorries": best_ok.sorries,
                    "conservative_sorries": best_ok.conservative_sorries,
                    "focus_goal_sig": best_ok.focus_goal_sig,
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
                    "sorries": picked.sorries,
                    "conservative_sorries": picked.conservative_sorries,
                    "focus_goal_sig": picked.focus_goal_sig,
                    "smt_hint": picked.smt_hint,
                    "rank_hint": picked.rank_hint,
                    "verify": {
                        "summary": picked.verify_summary,
                        "raw": if include_raw_verify { picked.verify_raw.clone().unwrap_or(serde_json::Value::Null) } else { serde_json::Value::Null }
                    }
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
                #[cfg(feature = "smt")]
                let (smt_ms_u64, smt_hits_u64, smt_misses_u64) = (prof_smt_ms, smt_cache_hits, smt_cache_misses);
                #[cfg(not(feature = "smt"))]
                let (smt_ms_u64, smt_hits_u64, smt_misses_u64) = (0u64, 0u64, 0u64);
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

            if let Some(p) = output_json {
                write_json(&p, &out)?;
                println!("{}", json!({"ok": true, "written": p.display().to_string()}).to_string());
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
                "suggestion": res.content,
                "raw": res.raw
            });

            if let Some(p) = output_json {
                write_json(&p, &out)?;
                println!(
                    "{}",
                    json!({"ok": true, "written": p.display().to_string()}).to_string()
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
                    json!({"ok": true, "written": p.display().to_string()}).to_string()
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
                    json!({"ok": true, "written": p.display().to_string()}).to_string()
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
                    let warnings = stdout.matches(": warning:").count()
                        + stderr.matches(": warning:").count();
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
                        json!({"ok": true, "written": p.display().to_string()}).to_string()
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
                        json!({"ok": true, "written": p.display().to_string()}).to_string()
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
                    //
                    // Use `axi`’s extractor to handle common failure modes:
                    // - ```json fenced blocks
                    // - leading/trailing commentary
                    // - “JSON-ish” output where the object/array is embedded in text
                    let review_struct =
                        axi::json_extract::extract_first_json_value(&r.content).ok();
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
                                let errors = row.get("errors").and_then(|x| x.as_u64()).unwrap_or(0);
                                if ok && errors == 0 {
                                    if let Some(f) = row.get("file").and_then(|x| x.as_str()) {
                                        ok_files.insert(f.to_string());
                                    }
                                }
                            }

                            if !ok_files.is_empty() {
                                if let Some(rs) = review_struct.as_mut() {
                                    // top_issues: drop “truncation/parse/compile” claims for ok files.
                                    if let Some(arr) = rs.get_mut("top_issues").and_then(|x| x.as_array_mut()) {
                                        let before = arr.len();
                                        arr.retain(|issue| {
                                            let title = issue.get("title").and_then(|x| x.as_str()).unwrap_or("");
                                            let detail = issue.get("detail").and_then(|x| x.as_str()).unwrap_or("");
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
                    json!({"ok": true, "written": p.display().to_string()}).to_string()
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
            let max_tool_iters = arg_u64(rest, "--max-tool-iters").unwrap_or(4) as usize;
            let tools = arg_value(rest, "--tools").unwrap_or_default();
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);

            let system_txt = if let Some(p) = system_file {
                fs::read_to_string(&p)
                    .map_err(|e| format!("read {}: {}", p.display(), e))?
            } else {
                system.unwrap_or_default()
            };
            let user_txt = if let Some(p) = user_file {
                fs::read_to_string(&p)
                    .map_err(|e| format!("read {}: {}", p.display(), e))?
            } else {
                user.unwrap_or_default()
            };
            if system_txt.trim().is_empty() && user_txt.trim().is_empty() {
                return Err("llm-chat requires --system/--system-file and/or --user/--user-file".to_string());
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
            let tools_mode = match tools {
                "" => "none",
                "agent" | "proofpatch" | "axi-proofpatch" => "agent",
                _ => {
                    return Err("llm-chat --tools must be: agent (or omit --tools for one-shot)".to_string());
                }
            };

            let out = if tools_mode == "agent" {
                let Some(rr) = repo_root.clone() else {
                    return Err("llm-chat --tools agent requires --repo <path>".to_string());
                };

                // Select provider/model using the same routing logic as the rest of proofpatch.
                let sel = rt
                    .block_on(plc::llm::select_provider_info(StdDuration::from_secs(3)))
                    .map_err(|e| format!("select provider: {e}"))?;

                // Build an axi model adapter.
                //
                // NOTE: We prefer the provider-specific OpenRouter adapter here because it adds the
                // headers OpenRouter expects and matches axi's tested request shape.
                let model: Box<dyn axi::agent::Model> = if sel.provider == "openrouter" {
                    let key = sel
                        .api_key
                        .clone()
                        .ok_or_else(|| "missing OPENROUTER_API_KEY".to_string())?;
                    Box::new(axi_providers::adapters::openrouter::OpenRouterAdapter::new(
                        key,
                        sel.model.clone(),
                    ))
                } else if sel.provider == "groq" {
                    let key = sel
                        .api_key
                        .clone()
                        .ok_or_else(|| "missing GROQ_API_KEY".to_string())?;
                    Box::new(axi_providers::adapters::groq::GroqAdapter::new(
                        key,
                        sel.model.clone(),
                    ))
                } else if sel.provider == "ollama" {
                    Box::new(axi_providers::adapters::ollama::OllamaAdapter::new(
                        // Ollama's OpenAI-compatible endpoint uses /v1.
                        format!("{}/v1", sel.base_url.trim_end_matches('/')),
                        sel.model.clone(),
                    ))
                } else {
                    let mut a = axi_providers::adapters::openai::GenericOpenAiAdapter::new(
                        sel.base_url.clone(),
                        sel.model.clone(),
                    );
                    if let Some(k) = sel.api_key.clone() {
                        a = a.with_api_key(k);
                    }
                    Box::new(a)
                };

                // Each tool closure needs its own owned copy of the repo root.
                // (DynamicTool stores the closure for the process lifetime.)
                let rr_ctx = rr.clone();
                let rr_verify = rr.clone();
                let rr_sorries = rr.clone();

                // Tool: proofpatch_context_pack
                let args_schema = schema_for!(ContextPackArgs);
                let tool = axi::tool::DynamicTool::new(
                    "proofpatch_context_pack",
                    "Return a bounded context pack (imports + excerpt + nearby decls) for a Lean file within the repo.",
                    args_schema,
                    move |raw_args: serde_json::Value| {
                        let args: ContextPackArgs = serde_json::from_value(raw_args)
                            .map_err(|e| axi::Error::ToolArgsInvalid(format!("proofpatch_context_pack: {e}")))?;
                        let line_1 = args.line.map(|x| x as usize);
                        let context_lines = args.context_lines as usize;
                        let nearby_lines = args.nearby_lines as usize;
                        let max_nearby = args.max_nearby as usize;
                        let max_imports = args.max_imports as usize;
                        let decl = args.decl.as_deref();
                        let pack = plc::build_context_pack(
                            &rr_ctx,
                            &args.file,
                            decl,
                            line_1,
                            context_lines,
                            nearby_lines,
                            max_nearby,
                            max_imports,
                        )
                        .map_err(|e| axi::Error::ToolExecutionFailed(e))?;
                        serde_json::to_value(pack)
                            .map_err(|e| axi::Error::ToolExecutionFailed(format!("json: {e}")))
                    },
                );

                // Tool: proofpatch_verify_summary
                let verify_args_schema = schema_for!(VerifySummaryArgs);
                let verify_tool = axi::tool::DynamicTool::new(
                    "proofpatch_verify_summary",
                    "Verify a Lean file and return a bounded summary (no raw stdout/stderr).",
                    verify_args_schema,
                    move |raw_args: serde_json::Value| {
                        let args: VerifySummaryArgs = serde_json::from_value(raw_args).map_err(|e| {
                            axi::Error::ToolArgsInvalid(format!("proofpatch_verify_summary: {e}"))
                        })?;
                        let rt = tokio::runtime::Runtime::new().map_err(|e| {
                            axi::Error::ToolExecutionFailed(format!("tokio runtime: {e}"))
                        })?;
                        let raw = rt
                            .block_on(plc::verify_lean_file(
                                &rr_verify,
                                &args.file,
                                StdDuration::from_secs(args.timeout_s),
                            ))
                            .map_err(axi::Error::ToolExecutionFailed)?;

                        let stdout = raw.stdout.as_str();
                        let stderr = raw.stderr.as_str();
                        let first_error_loc = plc::parse_first_error_loc(stdout, stderr)
                            .and_then(|loc| serde_json::to_value(loc).ok());

                        let errors =
                            stdout.matches(": error:").count() + stderr.matches(": error:").count();
                        let warnings = stdout.matches(": warning:").count()
                            + stderr.matches(": warning:").count();

                        Ok(json!({
                            "ok": raw.ok,
                            "timeout": raw.timeout,
                            "returncode": raw.returncode,
                            "counts": { "errors": errors, "warnings": warnings },
                            "first_error": stdout.lines().find(|l| l.contains(": error:"))
                                .or_else(|| stderr.lines().find(|l| l.contains(": error:"))),
                            "first_error_loc": first_error_loc,
                            "cmd": raw.cmd,
                            "cwd": raw.cwd,
                        }))
                    },
                );

                // Tool: proofpatch_locate_sorries
                let sorries_args_schema = schema_for!(LocateSorriesArgs);
                let sorries_tool = axi::tool::DynamicTool::new(
                    "proofpatch_locate_sorries",
                    "Locate `sorry`/`admit` occurrences in a file (bounded).",
                    sorries_args_schema,
                    move |raw_args: serde_json::Value| {
                        let args: LocateSorriesArgs = serde_json::from_value(raw_args).map_err(|e| {
                            axi::Error::ToolArgsInvalid(format!("proofpatch_locate_sorries: {e}"))
                        })?;
                        let locs = plc::locate_sorries_in_file(
                            &rr_sorries,
                            &args.file,
                            args.max_sorries as usize,
                            args.context_lines as usize,
                        )
                        .map_err(axi::Error::ToolExecutionFailed)?;
                        serde_json::to_value(locs)
                            .map_err(|e| axi::Error::ToolExecutionFailed(format!("json: {e}")))
                    },
                );

                // Tool: proofpatch_patch_first_sorry_in_decl
                let rr_patch = rr.clone();
                let patch_args_schema = schema_for!(PatchDeclArgs);
                let patch_tool = axi::tool::DynamicTool::new(
                    "proofpatch_patch_first_sorry_in_decl",
                    "Patch the first `sorry`/`admit` inside a declaration, optionally write to disk, optionally verify.",
                    patch_args_schema,
                    move |raw_args: serde_json::Value| {
                        let args: PatchDeclArgs = serde_json::from_value(raw_args).map_err(|e| {
                            axi::Error::ToolArgsInvalid(format!("proofpatch_patch_first_sorry_in_decl: {e}"))
                        })?;

                        let abs = rr_patch.join(&args.file);
                        if !abs.exists() {
                            return Err(axi::Error::ToolExecutionFailed(format!(
                                "File not found: {}",
                                abs.display()
                            )));
                        }
                        let original_text = std::fs::read_to_string(&abs).map_err(|e| {
                            axi::Error::ToolExecutionFailed(format!("read {}: {e}", abs.display()))
                        })?;

                        let patched = plc::patch_first_sorry_in_decl(
                            &original_text,
                            &args.decl,
                            &args.replacement,
                        )
                        .map_err(axi::Error::ToolExecutionFailed)?;

                        let lemma_still_contains_sorry =
                            plc::decl_block_contains_sorry(&patched.text, &args.decl)
                                .unwrap_or(true);

                        let mut wrote_path: Option<String> = None;
                        if args.write {
                            std::fs::write(&abs, patched.text.as_bytes()).map_err(|e| {
                                axi::Error::ToolExecutionFailed(format!("write {}: {e}", abs.display()))
                            })?;
                            wrote_path = Some(abs.display().to_string());
                        }

                        let verify_summary = if args.verify {
                            let rt = tokio::runtime::Runtime::new().map_err(|e| {
                                axi::Error::ToolExecutionFailed(format!("tokio runtime: {e}"))
                            })?;
                            let raw = if args.write {
                                rt.block_on(plc::verify_lean_file(
                                    &rr_patch,
                                    &args.file,
                                    StdDuration::from_secs(args.timeout_s),
                                ))
                                .map_err(axi::Error::ToolExecutionFailed)?
                            } else {
                                rt.block_on(plc::verify_lean_text(
                                    &rr_patch,
                                    &patched.text,
                                    StdDuration::from_secs(args.timeout_s),
                                ))
                                .map_err(axi::Error::ToolExecutionFailed)?
                            };

                            let stdout = raw.stdout.as_str();
                            let stderr = raw.stderr.as_str();
                            let first_error_loc = plc::parse_first_error_loc(stdout, stderr)
                                .and_then(|loc| serde_json::to_value(loc).ok());
                            let errors =
                                stdout.matches(": error:").count() + stderr.matches(": error:").count();
                            let warnings = stdout.matches(": warning:").count()
                                + stderr.matches(": warning:").count();
                            let first_error = stdout
                                .lines()
                                .find(|l| l.contains(": error:"))
                                .or_else(|| stderr.lines().find(|l| l.contains(": error:")))
                                .map(|s| truncate_str(s, 400));

                            Some(json!({
                                "ok": raw.ok,
                                "timeout": raw.timeout,
                                "returncode": raw.returncode,
                                "counts": { "errors": errors, "warnings": warnings },
                                "first_error": first_error,
                                "first_error_loc": first_error_loc,
                                "cmd": raw.cmd,
                                "cwd": raw.cwd,
                            }))
                        } else {
                            None
                        };

                        Ok(json!({
                            "file": args.file,
                            "decl": args.decl,
                            "write": args.write,
                            "written_file": wrote_path,
                            "patch": {
                                "changed": patched.changed,
                                "line": patched.line,
                                "before": truncate_str(&patched.before, 800),
                                "after": truncate_str(&patched.after, 800),
                                "indent": patched.indent,
                            },
                            "decl_still_contains_sorry": lemma_still_contains_sorry,
                            "verify_summary": verify_summary,
                        }))
                    },
                );

                let reasoning = effective_agent_reasoning_request(&sel.provider);
                let reasoning_out = reasoning.clone();

                // Agent config: bounded steps + best-effort wall clock budget (avoid runaway loops).
                let cfg = axi::AgentConfig {
                    max_steps: usize::max(3, max_tool_iters.saturating_mul(2) + 1),
                    max_wall_time_ms: Some(timeout_s.saturating_mul(1000)),
                    validate_schemas: true,
                    output_spec: Some(axi::constraints::OutputSpec::json_object()),
                    reasoning,
                    ..axi::AgentConfig::default()
                };

                let agent = axi::Agent::new((), system_txt.clone())
                    .with_config(cfg)
                    .add_invariant(axi::agent::ToolProtocolInvariant)
                    .add_tool(tool)
                    .add_tool(verify_tool)
                    .add_tool(sorries_tool)
                    .add_tool(patch_tool);

                let outcome = agent
                    .run::<serde_json::Value>(model.as_ref(), user_txt.clone(), None)
                    .map_err(|e| e.to_string());

                match outcome {
                    Ok(axi::RunOutcome::Completed(done)) => json!({
                        "ok": true,
                        "provider": sel.provider,
                        "model": sel.model,
                        "model_source": sel.model_source,
                        "model_env": sel.model_env,
                        "reasoning": reasoning_out,
                        "content": serde_json::to_string(&done.output).unwrap_or_else(|_| String::new()),
                        "content_struct": done.output,
                        "axi_trace": done.trace.as_json(),
                        "note": serde_json::Value::Null,
                    }),
                    Ok(axi::RunOutcome::Deferred(deferred)) => json!({
                        "ok": false,
                        "skipped": true,
                        "reason": "agent deferred (pending tool calls)",
                        "reasoning": reasoning_out,
                        "pending": deferred.pending.iter().map(|p| json!({"id": p.id, "name": p.name, "reason": p.reason, "arguments": p.arguments})).collect::<Vec<_>>(),
                        "axi_trace": deferred.state.trace.as_json(),
                    }),
                    Err(e) => {
                        if require_key {
                            return Err(format!("llm-chat failed: {e}"));
                        }
                        json!({"skipped": true, "reason": e})
                    }
                }
            } else {
                // Simple one-shot (existing non-agentic path).
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
                println!("{}", json!({"ok": true, "written": p.display().to_string()}).to_string());
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
                        let decl_kind = loc
                            .get("decl_kind")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let decl_name = loc
                            .get("decl_name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
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
            println!("{}", out.to_string());
            Ok(())
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
                    json!({"ok": true, "written": p.display().to_string()}).to_string()
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
                    json!({"ok": true, "written": p.display().to_string()}).to_string()
                );
            } else {
                println!("{}", out.to_string());
            }
            Ok(())
        }

        _ => Err(usage()),
    }
}
