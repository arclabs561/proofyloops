use serde_json::Value;

pub fn hash_text(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// Compute a stable-ish hash for a Lean goal dump entry (from `pp_dump`).
///
/// This is a lightweight stand-in for LeanTree-style factorized state hashing.
/// We hash:
/// - goal type (`pretty`)
/// - local context types (names ignored; order normalized)
///
/// Note: this is best-effort and intentionally backend-agnostic (plain strings).
pub fn hash_goal_sig(goal: &Value) -> Option<u64> {
    let ty = goal.get("pretty")?.as_str()?.trim();
    let mut parts: Vec<String> = Vec::new();
    if !ty.is_empty() {
        parts.push(format!("T:{ty}"));
    }
    if let Some(hyps) = goal.get("hyps").and_then(|v| v.as_array()) {
        let mut hs: Vec<String> = hyps
            .iter()
            .filter_map(|h| {
                if let Some(t) = h.get("type").and_then(|v| v.as_str()) {
                    Some(t.trim().to_string())
                } else if let Some(t) = h.get("text").and_then(|v| v.as_str()) {
                    Some(t.trim().to_string())
                } else {
                    None
                }
            })
            .filter(|s| !s.is_empty())
            .collect();
        hs.sort();
        hs.truncate(64);
        for h in hs {
            parts.push(format!("H:{h}"));
        }
    }
    if parts.is_empty() {
        return None;
    }
    Some(hash_text(&parts.join("\n")))
}

/// Hash a full proof state as a multiset of goal signatures (order-independent).
pub fn hash_state_key(pp_dump: &Value) -> Option<u64> {
    let goals = pp_dump.get("goals")?.as_array()?;
    let mut gs: Vec<u64> = goals.iter().filter_map(hash_goal_sig).collect();
    if gs.is_empty() {
        return None;
    }
    gs.sort();
    let mut s = String::new();
    for g in gs {
        s.push_str(&format!("{g}\n"));
    }
    Some(hash_text(&s))
}

pub fn default_det_candidates() -> Vec<String> {
    vec![
        // Cheap first.
        "by\n  (simp; done)".to_string(),
        "by\n  classical\n  (simp; done)".to_string(),
        // Heavier automation.
        "by\n  aesop".to_string(),
        "by\n  classical\n  aesop".to_string(),
        "by\n  (simp_all; done)".to_string(),
        "by\n  (omega; done)".to_string(),
        "by\n  (nlinarith; done)".to_string(),
        "by\n  (linarith; done)".to_string(),
        "by\n  (ring_nf; done)".to_string(),
        "by\n  (norm_num; done)".to_string(),
    ]
}

pub fn parse_json_string_array(s: &str) -> Option<Vec<String>> {
    fn parse_value(v: &Value) -> Option<Vec<String>> {
        let xs = v.as_array()?;
        let mut out = Vec::new();
        for x in xs {
            if let Some(ss) = x.as_str() {
                let t = ss.trim();
                if !t.is_empty() {
                    out.push(t.to_string());
                }
            }
        }
        if out.is_empty() { None } else { Some(out) }
    }

    // Fast path: entire string is a JSON array.
    if let Ok(v) = serde_json::from_str::<Value>(s) {
        if let Some(xs) = parse_value(&v) {
            return Some(xs);
        }
    }

    // Common LLM behavior: wrap JSON in markdown fences or extra text. Reuse `axi`’s
    // bounded extractor to recover the first JSON value, then check whether it is a
    // string array.
    if let Ok(v) = axi::json_extract::extract_first_json_value(s) {
        if let Some(xs) = parse_value(&v) {
            return Some(xs);
        }
    }

    None
}

pub fn sanitize_candidates(mut xs: Vec<String>) -> Vec<String> {
    // Keep this bounded and fairly strict: huge candidates are usually junk.
    xs.retain(|s| !s.trim().is_empty());
    xs.truncate(24);
    xs.retain(|s| s.chars().count() <= 4_000);
    // Deduplicate while preserving order.
    let mut seen = std::collections::HashSet::new();
    xs.retain(|s| seen.insert(s.clone()));
    xs
}

pub fn is_made_no_progress(first_error: Option<&str>) -> bool {
    first_error
        .unwrap_or("")
        .to_ascii_lowercase()
        .contains("made no progress")
}

pub fn adapt_candidates_for_error(base: &[String], first_error: Option<&str>) -> Vec<String> {
    let mut out = base.to_vec();
    let err = first_error.unwrap_or("").to_lowercase();

    // If automation made no progress, widen the surface with suggestion tactics.
    if err.contains("made no progress") {
        // Prefer *real* tactics; Mathlib `...?` suggestion tactics can hide admitted proofs
        // (they often end with a synthetic `sorry`).
        out.push("by\n  (simp; done)".to_string());
        out.push("by\n  aesop".to_string());
        out.push("by\n  classical\n  (simp; done)".to_string());
        out.push("by\n  classical\n  aesop".to_string());
    }

    // Some goals need classical instances for automation.
    if err.contains("failed to synthesize") && err.contains("decidable") {
        out.push("by\n  classical\n  aesop".to_string());
        out.push("by\n  classical\n  (simp; done)".to_string());
    }

    // If a tactic is unknown, avoid leaning harder into suggestion tactics.
    if err.contains("unknown tactic") {
        out.retain(|c| !c.contains("?"));
    }

    sanitize_candidates(out)
}

fn strip_leading_by(candidate: &str) -> String {
    let s = candidate.trim();
    if s == "by" {
        return String::new();
    }
    if let Some(rest) = s.strip_prefix("by\n") {
        // Drop one common indentation level across the whole block.
        let lines: Vec<&str> = rest.lines().collect();
        let mut out: Vec<String> = Vec::new();
        for (i, ln) in lines.iter().enumerate() {
            if i == 0 {
                out.push(ln.strip_prefix("  ").unwrap_or(ln).to_string());
            } else {
                out.push(ln.strip_prefix("  ").unwrap_or(ln).to_string());
            }
        }
        return out.join("\n");
    }
    if let Some(rest) = s.strip_prefix("by ") {
        return rest.to_string();
    }
    candidate.to_string()
}

/// If the selected `sorry` is in a `by`-tactic position, candidates should be tactic scripts
/// like `simp`/`aesop` rather than `by\n  simp`.
pub fn adapt_candidates_for_sorry_line(base: &[String], line_text: &str) -> Vec<String> {
    let lt = line_text.trim();
    // Important: a bare `sorry` line is ambiguous: it could be in a term position:
    //
    //   theorem foo : P :=
    //     sorry
    //
    // In that case replacing `sorry` with `simp` is a type error (term elaboration),
    // while `by\n  simp` remains a valid term.
    //
    // We only strip `by` for the unambiguous inline pattern `by sorry` / `by admit`.
    let is_tactic_line = lt.contains("by sorry") || lt.contains("by admit");
    if !is_tactic_line {
        return base.to_vec();
    }

    let mut out: Vec<String> = base.iter().map(|c| strip_leading_by(c)).collect();
    out.retain(|s| !s.trim().is_empty());
    sanitize_candidates(out)
}

/// Strip `by` prefixes when the hole is known to be a tactic hole (inside a `by` block).
pub fn adapt_candidates_for_tactic_hole(base: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for c in base {
        let stripped = strip_leading_by(c);
        let t = stripped.trim();
        if t.is_empty() {
            continue;
        }
        // For single-line tactics that commonly "succeed" without closing goals (e.g. `simp`),
        // enforce a close-or-fail shape to avoid `unsolved goals` downstream.
        //
        // For multi-line blocks (e.g. oracle skeletons), keep them as-is.
        if !t.contains('\n') && !t.contains("done") && !t.contains(';') {
            out.push(format!("({t}; done)"));
        } else {
            out.push(t.to_string());
        }
    }
    sanitize_candidates(out)
}

/// Context-aware adapter for `sorry` replacement candidates.
pub fn adapt_candidates_for_sorry_context(
    base: &[String],
    line_text: &str,
    is_tactic_context: bool,
) -> Vec<String> {
    if is_tactic_context {
        return adapt_candidates_for_tactic_hole(base);
    }
    adapt_candidates_for_sorry_line(base, line_text)
}

/// Extract a Lean "Initial goal:" block (best-effort) from tactic failure output.
///
/// Example (from `aesop` failures):
/// ```text
/// error: tactic 'aesop' failed, made no progress
/// Initial goal:
///   ...
///   ⊢ ...
/// ```
pub fn extract_initial_goal_block(text: &str) -> Option<String> {
    let mut lines = text.lines();
    // Find the marker line first.
    while let Some(l) = lines.next() {
        if l.trim() == "Initial goal:" {
            // Collect following lines until a blank line or a new diagnostic.
            let mut out = String::new();
            out.push_str("Initial goal:\n");
            let mut n_lines = 0usize;
            for l2 in &mut lines {
                let t = l2.trim_end();
                if t.trim().is_empty() {
                    break;
                }
                if t.contains(": error:") || t.contains(": error(") || t.contains(": warning:") || t.contains(": warning(") {
                    break;
                }
                out.push_str(t);
                out.push('\n');
                n_lines += 1;
                if n_lines >= 80 || out.chars().count() >= 6_000 {
                    break;
                }
            }
            return Some(out);
        }
    }
    None
}

pub fn verify_score_key(summary: &Value, sorries: usize, conservative: usize) -> (i32, i64, i64, i64) {
    let ok = summary.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
    let errors = summary
        .get("counts")
        .and_then(|c| c.get("errors"))
        .and_then(|v| v.as_u64())
        .unwrap_or(999) as i64;
    let sorry_warnings = summary
        .get("counts")
        .and_then(|c| c.get("sorry_warnings"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as i64;
    let warnings = summary
        .get("counts")
        .and_then(|c| c.get("warnings"))
        .and_then(|v| v.as_u64())
        .unwrap_or(999) as i64;
    let made_no_progress = summary
        .get("first_error")
        .and_then(|v| v.as_str())
        .map(|s| is_made_no_progress(Some(s)))
        .unwrap_or(false);
    // Penalize “no progress” errors as worst-case (they indicate we’re wasting tries).
    let np_penalty = if made_no_progress { 1 } else { 0 };
    // Sort best-first:
    // - ok first
    // - fewer errors, then fewer synthetic-sorry warnings, then fewer `locate` sorries,
    //   then fewer conservative sorries, then fewer warnings
    let ok_key = if ok { 0 } else { 1 };
    (ok_key, errors + np_penalty + (sorry_warnings * 10), sorries as i64, conservative as i64 + warnings)
}

pub fn progress_score_key(summary: &Value, sorries: usize, conservative: usize) -> (i64, i32, i64, i64) {
    // Prefer fewer remaining sorries; use errors as a tiebreak.
    let ok = summary.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
    let ok_key = if ok { 0 } else { 1 };
    let sorry_warnings = summary
        .get("counts")
        .and_then(|c| c.get("sorry_warnings"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as i64;
    let errors = summary
        .get("counts")
        .and_then(|c| c.get("errors"))
        .and_then(|v| v.as_u64())
        .unwrap_or(999) as i64;
    // Key: fewer synthetic-sorry warnings first, then fewer explicit sorries, then ok/errors.
    (sorry_warnings, ok_key, sorries as i64, errors + conservative as i64)
}

