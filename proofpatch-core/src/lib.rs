//! `proofpatch-core`: small utilities for Lean 4 proof work.
//!
//! Scope:
//! - run `lake env lean` to verify files or temporary excerpts
//! - locate `sorry` regions and build bounded “prompt packs”
//! - optional LLM calls via an OpenAI-compatible API (Ollama/Groq/OpenAI/OpenRouter)
//!
//! Output discipline:
//! - keep outputs JSON-friendly (`serde` types)
//! - keep any “review” payload bounded (size caps + secret redaction)
//!
//! Entrypoints:
//! - the CLI binary lives in `proofpatch-core/src/bin/proofpatch.rs`
//! - the MCP server wrapper lives in `proofpatch/mcp-server`
//!
//! Environment:
//! - Prefer `PROOFPATCH_*`.
//! - LLM routing can use:
//!   - `OLLAMA_MODEL` (+ optional `OLLAMA_HOST`)
//!   - `GROQ_API_KEY` and `GROQ_MODEL`
//!   - `OPENAI_API_KEY` and `OPENAI_MODEL` (+ optional `OPENAI_BASE_URL`)
//!   - `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` (+ optional `OPENROUTER_BASE_URL`)
//!
use regex::Regex;
use serde::{Deserialize, Serialize};
// (no extra imports needed for LSP backend)
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;
use tokio::process::Command;

pub mod llm;
#[cfg(feature = "planner")]
pub mod planner;
pub mod review;
pub mod tree_search;
#[cfg(feature = "lsp")]
mod lsp_client;

#[derive(Debug, Clone)]
struct LeanEnv {
    /// Environment variables captured from `lake env env` (or equivalent).
    env: HashMap<String, String>,
}

static LEAN_ENV_CACHE: OnceLock<Mutex<HashMap<PathBuf, LeanEnv>>> = OnceLock::new();

async fn get_or_compute_lean_env(repo_root: &Path, timeout_s: Duration) -> Option<LeanEnv> {
    let cache = LEAN_ENV_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(guard) = cache.lock() {
        if let Some(v) = guard.get(repo_root) {
            return Some(v.clone());
        }
    }

    let lake = resolve_lake();
    let mut cmd = Command::new(&lake);
    cmd.arg("env").arg("env").current_dir(repo_root);
    maybe_extend_lean_path_for_lake_env(&mut cmd);

    let out = tokio::time::timeout(timeout_s, cmd.output())
        .await
        .ok()?
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let mut env: HashMap<String, String> = HashMap::new();
    for line in stdout.lines() {
        let t = line.trim_end();
        if t.is_empty() {
            continue;
        }
        if let Some((k, v)) = t.split_once('=') {
            let k = k.trim();
            if k.is_empty() {
                continue;
            }
            env.insert(k.to_string(), v.to_string());
        }
    }
    if env.is_empty() {
        return None;
    }

    let v = LeanEnv { env };
    if let Ok(mut guard) = cache.lock() {
        guard.insert(repo_root.to_path_buf(), v.clone());
    }
    Some(v)
}

async fn run_lean_with_env(
    repo_root: &Path,
    lean_args: &[String],
    timeout_s: Duration,
) -> Result<(bool, bool, Option<i32>, String, String), String> {
    let Some(lean_env) = get_or_compute_lean_env(repo_root, timeout_s).await else {
        return Err("no lean env available".to_string());
    };
    let mut cmd = Command::new("lean");
    cmd.current_dir(repo_root);
    cmd.env_clear();
    cmd.envs(lean_env.env);
    for a in lean_args {
        cmd.arg(a);
    }
    let out = tokio::time::timeout(timeout_s, cmd.output())
        .await
        .map_err(|_| "timeout".to_string());
    let (ok, timeout, returncode, stdout, stderr) = match out {
        Err(_) => (false, true, None, String::new(), String::new()),
        Ok(Err(e)) => (
            false,
            false,
            None,
            String::new(),
            format!("failed to execute: {}", e),
        ),
        Ok(Ok(output)) => {
            let ok = output.status.success();
            let returncode = output.status.code();
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            (ok, false, returncode, stdout, stderr)
        }
    };
    Ok((ok, timeout, returncode, stdout, stderr))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchSource {
    pub url: String,
    /// Best-effort canonicalized URL used for deduping (e.g. arXiv pdf -> abs).
    pub canonical_url: Option<String>,
    pub title: Option<String>,
    pub snippet: Option<String>,
    pub origin: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchNotes {
    pub raw_urls: usize,
    pub deduped_urls: usize,
    pub sources: Vec<ResearchSource>,
}

fn canonicalize_url(url: &str) -> Option<String> {
    let u = url.trim().trim_end_matches('/');
    // arXiv: treat pdf and abs as the same paper.
    // - https://arxiv.org/pdf/1234.5678.pdf  -> https://arxiv.org/abs/1234.5678
    // - https://arxiv.org/pdf/1234.5678v2.pdf -> https://arxiv.org/abs/1234.5678v2
    for host in ["https://arxiv.org/abs/", "http://arxiv.org/abs/"] {
        if let Some(rest) = u.strip_prefix(host) {
            let id = rest.trim_matches('/');
            if !id.is_empty() {
                return Some(format!("https://arxiv.org/abs/{id}"));
            }
        }
    }
    for host in ["https://arxiv.org/pdf/", "http://arxiv.org/pdf/"] {
        if let Some(rest) = u.strip_prefix(host) {
            if let Some(id) = rest.strip_suffix(".pdf") {
                let id = id.trim_matches('/');
                if !id.is_empty() {
                    return Some(format!("https://arxiv.org/abs/{id}"));
                }
            }
        }
    }
    None
}

fn is_urlish_key(k: &str) -> bool {
    matches!(
        k,
        "url"
            | "link"
            | "href"
            | "pdf_url"
            | "pdfUrl"
            | "arxiv_url"
            | "arxivUrl"
            | "paper_url"
            | "paperUrl"
    )
}

fn pick_string_field<'a>(obj: &'a serde_json::Map<String, serde_json::Value>, keys: &[&str]) -> Option<&'a str> {
    for k in keys {
        if let Some(v) = obj.get(*k).and_then(|v| v.as_str()) {
            if !v.trim().is_empty() {
                return Some(v);
            }
        }
    }
    None
}

fn should_ignore_url(url: &str) -> bool {
    // Avoid schema/self-referential noise.
    url.contains("json-schema.org")
        || url.contains("schemas.cursor")
        || url.contains("localhost")
        || url.starts_with("file:")
}

fn collect_research_sources(
    v: &serde_json::Value,
    out: &mut Vec<ResearchSource>,
    seen: &mut HashSet<String>,
    raw_urls: &mut usize,
    current_origin: Option<&str>,
) {
    match v {
        serde_json::Value::Array(xs) => {
            for x in xs {
                collect_research_sources(x, out, seen, raw_urls, current_origin);
            }
        }
        serde_json::Value::Object(obj) => {
            // Track origin as we descend (tool/server context often wraps result arrays).
            let mut origin_here: Option<String> = current_origin.map(|s| s.to_string());
            let server = obj.get("server").and_then(|v| v.as_str());
            let tool_name = obj
                .get("toolName")
                .and_then(|v| v.as_str())
                .or_else(|| obj.get("tool").and_then(|v| v.as_str()))
                .or_else(|| obj.get("source").and_then(|v| v.as_str()));
            if let (Some(srv), Some(tn)) = (server, tool_name) {
                origin_here = Some(format!("{srv}:{tn}"));
            } else if let Some(tn) = tool_name {
                origin_here = Some(tn.to_string());
            }

            // If this object directly contains a URL field, emit a source.
            for (k, vv) in obj.iter() {
                if !is_urlish_key(k) {
                    continue;
                }
                let Some(url) = vv.as_str() else {
                    continue;
                };
                let url = url.trim();
                if url.is_empty() || !url.starts_with("http") || should_ignore_url(url) {
                    continue;
                }
                *raw_urls += 1;
                let canonical = canonicalize_url(url);
                let key = canonical.as_deref().unwrap_or(url).to_string();
                if seen.insert(key) {
                    let title = pick_string_field(obj, &["title", "name"]).map(|s| s.to_string());
                    let snippet = pick_string_field(obj, &["snippet", "summary", "content", "text", "abstract"])
                        .map(|s| s.to_string());
                    let origin = pick_string_field(obj, &["origin", "source", "server", "tool", "toolName"])
                        .map(|s| s.to_string())
                        .or_else(|| origin_here.clone());
                    out.push(ResearchSource {
                        url: url.to_string(),
                        canonical_url: canonical,
                        title,
                        snippet,
                        origin,
                    });
                }
            }

            // Recurse into all children.
            for vv in obj.values() {
                collect_research_sources(
                    vv,
                    out,
                    seen,
                    raw_urls,
                    origin_here.as_deref().or(current_origin),
                );
            }
        }
        _ => {}
    }
}

pub fn ingest_research_json(v: &serde_json::Value) -> ResearchNotes {
    let mut sources = Vec::new();
    let mut seen = HashSet::new();
    let mut raw_urls = 0usize;
    collect_research_sources(v, &mut sources, &mut seen, &mut raw_urls, None);
    ResearchNotes {
        raw_urls,
        deduped_urls: sources.len(),
        sources,
    }
}

fn is_stopword(s: &str) -> bool {
    matches!(
        s,
        // ultra-common English
        "the" | "and" | "for" | "with" | "that" | "this" | "from" | "into" | "then" | "than"
            | "also" | "only" | "just" | "some" | "more" | "most" | "when" | "where" | "what"
            | "which" | "while" | "will" | "should" | "would" | "could"
            // Lean/common proof noise
            | "lean" | "mathlib" | "proof" | "lemma" | "theorem" | "def" | "have" | "intro"
            | "exact" | "simp" | "cases" | "case" | "by" | "sorry" | "admit"
            | "file" | "line" | "col" | "warning" | "error"
            // path-ish noise (common in diagnostics/logs)
            // NOTE: do not include user-specific identifiers here.
            | "users" | "documents" | "downloads" | "desktop" | "home" | "tmp" | "private" | "var"
    )
}

fn tokenize(s: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() {
            cur.push(ch.to_ascii_lowercase());
        } else if !cur.is_empty() {
            if cur.len() >= 3 {
                let tok = std::mem::take(&mut cur);
                if !is_stopword(&tok) {
                    out.insert(tok);
                }
            } else {
                cur.clear();
            }
        }
    }
    if cur.len() >= 3 {
        if !is_stopword(&cur) {
            out.insert(cur);
        }
    }
    out
}

fn overlap_and_jaccard(a: &HashSet<String>, b: &HashSet<String>) -> (usize, f32) {
    if a.is_empty() || b.is_empty() {
        return (0, 0.0);
    }
    let mut inter = 0usize;
    for x in a {
        if b.contains(x) {
            inter += 1;
        }
    }
    let union = a.len() + b.len() - inter;
    if union == 0 {
        (inter, 0.0)
    } else {
        (inter, (inter as f32) / (union as f32))
    }
}

/// Attach `top_k` relevant sources to each `next_actions[*]` entry in a report-like JSON payload.
///
/// This is intentionally heuristic and inspectable:
/// - score = Jaccard(tokenize(action_text), tokenize(source_text))
/// - action_text uses: decl_name + excerpt + any embedded `query` strings from the research plan
/// - source_text uses: url + title + snippet
pub fn attach_research_matches_to_next_actions(
    report_json: &mut serde_json::Value,
    notes: &ResearchNotes,
    top_k: usize,
) {
    let Some(actions) = report_json
        .get_mut("next_actions")
        .and_then(|v| v.as_array_mut())
    else {
        return;
    };

    for a in actions.iter_mut() {
        let mut action_text = String::new();
        if let Some(decl) = a.get("decl_name").and_then(|v| v.as_str()) {
            action_text.push_str(decl);
            action_text.push('\n');
        }
        if let Some(ex) = a.get("excerpt").and_then(|v| v.as_str()) {
            action_text.push_str(ex);
            action_text.push('\n');
        }
        // Walk research.plan.calls[*].arguments.query if present.
        if let Some(calls) = a
            .get("research")
            .and_then(|v| v.get("plan"))
            .and_then(|v| v.get("calls"))
            .and_then(|v| v.as_array())
        {
            for c in calls {
                if let Some(q) = c
                    .get("arguments")
                    .and_then(|v| v.get("query"))
                    .and_then(|v| v.as_str())
                {
                    action_text.push_str(q);
                    action_text.push('\n');
                }
            }
        }

        let action_tokens = tokenize(&action_text);

        let mut scored: Vec<(usize, f32, &ResearchSource)> = notes
            .sources
            .iter()
            .map(|s| {
                let mut src_text = String::new();
                src_text.push_str(&s.url);
                src_text.push('\n');
                if let Some(t) = &s.title {
                    src_text.push_str(t);
                    src_text.push('\n');
                }
                if let Some(sn) = &s.snippet {
                    src_text.push_str(sn);
                    src_text.push('\n');
                }
                let src_tokens = tokenize(&src_text);
                let (inter, score) = overlap_and_jaccard(&action_tokens, &src_tokens);
                (inter, score, s)
            })
            .collect();

        scored.sort_by(|(ia, sa, a), (ib, sb, b)| {
            ib.cmp(ia)
                .then_with(|| sb.partial_cmp(sa).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.url.cmp(&b.url))
        });

        let matches: Vec<serde_json::Value> = scored
            .into_iter()
            // Require at least 2 non-trivial shared tokens to avoid pure-noise matches.
            .filter(|(inter, _s, _)| *inter >= 2)
            .take(top_k)
            .map(|(inter, score, s)| {
                serde_json::json!({
                    "overlap": inter,
                    "score": score,
                    "url": s.url,
                    "canonical_url": s.canonical_url,
                    "title": s.title,
                    "origin": s.origin,
                })
            })
            .collect();

        if let Some(obj) = a.as_object_mut() {
            obj.insert("research_matches".to_string(), serde_json::Value::Array(matches));
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyResult {
    pub ok: bool,
    pub timeout: bool,
    pub returncode: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub cmd: Vec<String>,
    pub cwd: String,
    pub tmp_file: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticLoc {
    pub path: String,
    pub line: usize,
    pub col: usize,
    pub kind: String,
}

/// Parse the first Lean diagnostic location from `stdout`/`stderr` (best-effort).
///
/// Typical Lean format:
/// - `/abs/path/File.lean:276:8: error: ...`
/// - `File.lean:276:8: error: ...`
///
/// We only return the first `error:` line (warnings are intentionally ignored).
pub fn parse_first_error_loc(stdout: &str, stderr: &str) -> Option<DiagnosticLoc> {
    // Parse a Lean diagnostic location line.
    //
    // Typical formats:
    // - `/abs/path/File.lean:276:8: error: ...`
    // - `File.lean:276:8: error: ...`
    //
    // Windows paths include drive letters (`C:\...`) which contain a `:`; regexes that
    // naively split on the first `:` will mis-parse them. Prefer a right-to-left parse.
    fn parse_line(line: &str) -> Option<(String, usize, usize)> {
        let idx = line.find(": error:").or_else(|| line.find(": error("))?;
        let prefix = line[..idx].trim_end();
        // prefix should end with `:<line>:<col>`; parse from the right.
        let mut it = prefix.rsplitn(3, ':');
        let col_s = it.next()?;
        let line_s = it.next()?;
        let path = it.next()?.to_string();
        let line_no = line_s.trim().parse::<usize>().ok()?;
        let col_no = col_s.trim().parse::<usize>().ok()?;
        Some((path, line_no, col_no))
    }

    fn is_error_line(l: &str) -> bool {
        l.contains(": error:") || l.contains(": error(")
    }

    // Find the first error line (prefer stdout then stderr).
    let first = stdout
        .lines()
        .find(|l| is_error_line(l))
        .or_else(|| stderr.lines().find(|l| is_error_line(l)))?;
    let (path, line, col) = parse_line(first)?;
    Some(DiagnosticLoc {
        path,
        line,
        col,
        kind: "error".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_first_error_loc_abs_path() {
        let out = "/tmp/Foo.lean:12:34: error: boom\nmore";
        let loc = parse_first_error_loc(out, "").expect("expected loc");
        assert_eq!(loc.path, "/tmp/Foo.lean");
        assert_eq!(loc.line, 12);
        assert_eq!(loc.col, 34);
        assert_eq!(loc.kind, "error");
    }

    #[test]
    fn test_parse_first_error_loc_windows_drive_path() {
        let out = r"C:\work\Foo.lean:12:34: error: boom";
        let loc = parse_first_error_loc(out, "").expect("expected loc");
        assert_eq!(loc.path, r"C:\work\Foo.lean");
        assert_eq!(loc.line, 12);
        assert_eq!(loc.col, 34);
        assert_eq!(loc.kind, "error");
    }

    #[test]
    fn test_parse_first_error_loc_rel_path() {
        let out = "Foo.lean:1:2: error: boom";
        let loc = parse_first_error_loc(out, "").expect("expected loc");
        assert_eq!(loc.path, "Foo.lean");
        assert_eq!(loc.line, 1);
        assert_eq!(loc.col, 2);
    }

    #[test]
    fn test_parse_first_error_loc_accepts_error_code_paren() {
        let out = "Foo.lean:9:8: error(lean.unknownIdentifier): Unknown identifier `simp`";
        let loc = parse_first_error_loc(out, "").expect("expected loc");
        assert_eq!(loc.path, "Foo.lean");
        assert_eq!(loc.line, 9);
        assert_eq!(loc.col, 8);
        assert_eq!(loc.kind, "error");
    }

    #[test]
    fn test_parse_first_error_loc_ignores_warnings() {
        let out = "Foo.lean:1:2: warning: nope\nFoo.lean:3:4: error: yes";
        let loc = parse_first_error_loc(out, "").expect("expected loc");
        assert_eq!(loc.line, 3);
        assert_eq!(loc.col, 4);
    }

    #[test]
    fn test_extract_line_span_basic() {
        let txt = "a\nb\nc\nd\ne\n";
        let lines: Vec<&str> = txt.lines().collect();
        let (start, end, excerpt) = extract_line_span(&lines, 3, 1);
        assert_eq!(start, 2);
        assert_eq!(end, 4);
        assert_eq!(excerpt, "b\nc\nd");
    }

    #[test]
    fn conservative_sorry_count_includes_comments_but_locator_ignores() {
        let txt = r#"
-- sorry in comment (should be counted conservatively)
def foo : Nat := 1
"#;
        let conservative = count_sorry_tokens_conservative(txt).expect("count");
        assert_eq!(conservative, 1);
        let locs = locate_sorries_in_text(txt, 50, 2).expect("locate");
        assert_eq!(locs.len(), 0);
    }

    #[test]
    fn select_primary_sorry_picks_closest_to_error_line_else_first() {
        let txt = "def a : Nat := by\n  sorry\n\ndef b : Nat := by\n  sorry\n";
        let locs = locate_sorries_in_text(txt, 50, 1).expect("locs");
        assert_eq!(locs.len(), 2);

        // No error line → first sorry in source order.
        let s0 = select_primary_sorry(None, &locs).expect("primary");
        assert_eq!(s0.line, locs[0].line);

        // Error closer to second sorry.
        let s1 = select_primary_sorry(Some(locs[1].line), &locs).expect("primary");
        assert_eq!(s1.line, locs[1].line);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchResult {
    pub text: String,
    pub changed: bool,
    pub line: usize,
    pub indent: String,
    pub before: String,
    pub after: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptPayload {
    pub repo_root: String,
    pub file: String,
    pub decl: String,
    pub excerpt: String,
    pub system: String,
    pub user: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SorryLocation {
    /// Matched token: `sorry` or `admit`.
    pub token: String,
    /// Nearest enclosing declaration kind (best-effort): theorem|lemma|def|instance|abbrev|structure|class.
    pub decl_kind: Option<String>,
    /// Nearest enclosing declaration name (best-effort).
    pub decl_name: Option<String>,
    /// 1-based line number of the enclosing declaration header (best-effort).
    pub decl_line: Option<usize>,
    /// 1-based line number.
    pub line: usize,
    /// 1-based column number (byte-based within the line; for display only).
    pub col: usize,
    /// The full line text containing the match.
    pub line_text: String,
    /// Suggested patch region start line (1-based, inclusive).
    pub region_start: usize,
    /// Suggested patch region end line (1-based, inclusive).
    pub region_end: usize,
    /// Small local excerpt around the line (for humans/agents).
    pub excerpt: String,
}

pub fn parse_dotenv(path: &Path) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let Ok(text) = std::fs::read_to_string(path) else {
        return out;
    };
    for raw in text.lines() {
        let mut line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("export ") {
            line = rest.trim_start();
        }
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let k = k.trim();
        if k.is_empty() {
            continue;
        }
        let mut v = v.trim().to_string();
        if v.len() >= 2 {
            let bytes = v.as_bytes();
            let first = bytes[0];
            let last = bytes[bytes.len() - 1];
            if first == last && (first == b'"' || first == b'\'') {
                v = v[1..v.len() - 1].to_string();
            }
        }
        // Never override existing env vars.
        if std::env::var(k).ok().as_deref().unwrap_or("").is_empty() {
            out.insert(k.to_string(), v);
        }
    }
    out
}

pub fn load_dotenv_if_present(repo_root: &Path) {
    let p = repo_root.join(".env");
    for (k, v) in parse_dotenv(&p) {
        if std::env::var(&k).ok().as_deref().unwrap_or("").is_empty() {
            std::env::set_var(k, v);
        }
    }
}

fn env_truthy(name: &str, default_on: bool) -> bool {
    let v = std::env::var(name).ok().unwrap_or_default();
    let v = v.trim().to_lowercase();
    if v.is_empty() {
        return default_on;
    }
    !matches!(v.as_str(), "0" | "false" | "no" | "off")
}

fn platform_path_sep() -> &'static str {
    if cfg!(windows) {
        ";"
    } else {
        ":"
    }
}

fn maybe_extend_lean_path_for_lake_env(cmd: &mut Command) {
    let extra = std::env::var("PROOFPATCH_EXTRA_LEAN_PATH")
        .ok()
        .unwrap_or_default()
        .trim()
        .to_string();
    if extra.is_empty() {
        return;
    }

    // `lake env` will set up a LEAN_PATH for the target repo; we additionally allow
    // prefixing/suffixing paths so helper oleans (e.g. ProofpatchTools) can be resolved.
    //
    // This is intentionally env-driven (opt-in) to avoid surprising behavior.
    let existing = std::env::var("LEAN_PATH").ok().unwrap_or_default();
    let sep = platform_path_sep();
    let merged = if existing.trim().is_empty() {
        extra
    } else {
        format!("{}{}{}", existing, sep, extra)
    };
    cmd.env("LEAN_PATH", merged);
}

fn has_any_llm_key() -> bool {
    // We keep this narrow: these are the most common OpenAI-compatible key envs.
    // (We do NOT print or log them anywhere.)
    let ok = |k: &str| std::env::var(k).ok().as_deref().unwrap_or("").trim().len() > 0;
    ok("OPENROUTER_API_KEY") || ok("OPENAI_API_KEY") || ok("GROQ_API_KEY")
}

fn looks_like_missing_olean(stdout: &str, stderr: &str) -> bool {
    let s = format!("{stdout}\n{stderr}");
    s.contains("object file '") && s.contains(".olean") && s.contains("does not exist")
}

/// Merge `mcpServers.*.env` blocks from Cursor's `mcp.json` into process env.
/// Never overrides existing env vars.
///
/// Controls:
/// - `PROOFPATCH_MCP_JSON_PATH`: override path to mcp.json (useful for tests)
pub fn load_cursor_mcp_env_if_present() {
    let p = std::env::var("PROOFPATCH_MCP_JSON_PATH")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".cursor")
                .join("mcp.json")
        });

    let Ok(txt) = std::fs::read_to_string(&p) else {
        return;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) else {
        return;
    };
    let Some(servers) = v.get("mcpServers").and_then(|x| x.as_object()) else {
        return;
    };
    for (_, cfg) in servers.iter() {
        let Some(env) = cfg.get("env").and_then(|x| x.as_object()) else {
            continue;
        };
        for (k, vv) in env.iter() {
            let Some(vs) = vv.as_str() else {
                continue;
            };
            if std::env::var(k).ok().as_deref().unwrap_or("").is_empty() {
                std::env::set_var(k, vs);
            }
        }
    }
}

/// Load dotenv with covolume-style “super-workspace” convenience:
/// - load `<repo_root>/.env`
/// - if still no API keys, optionally scan sibling dirs (one-level deep) for a `.env`
///   containing `OPENROUTER_API_KEY` or `OPENAI_API_KEY`
///
/// Controls (all optional):
/// - `PROOFPATCH_DOTENV_SEARCH` (default: on): set to 0/false/off to disable
/// - `PROOFPATCH_DOTENV_SEARCH_ROOT` (default: repo_root.parent): override search root
pub fn load_dotenv_smart(repo_root: &Path) {
    // Base: repo-local .env
    load_dotenv_if_present(repo_root);

    // Local convenience: Cursor MCP env blocks (never override).
    load_cursor_mcp_env_if_present();

    if has_any_llm_key() {
        return;
    }

    if !env_truthy("PROOFPATCH_DOTENV_SEARCH", true) {
        return;
    }

    let search_root = std::env::var("PROOFPATCH_DOTENV_SEARCH_ROOT")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root.parent().unwrap_or(repo_root).to_path_buf());

    // Super-workspace convenience: also try the search_root's own `.env` (e.g. `~/dev/.env`)
    // before scanning sibling directories. This matches the expectation of keeping a single
    // workspace-wide key file.
    load_dotenv_if_present(&search_root);
    if has_any_llm_key() {
        return;
    }

    let Ok(entries) = std::fs::read_dir(&search_root) else {
        return;
    };
    let mut dirs = Vec::new();
    for e in entries.flatten() {
        let p = e.path();
        if !p.is_dir() {
            continue;
        }
        if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
            if name.starts_with('.') {
                continue;
            }
        }
        if p == repo_root {
            continue;
        }
        dirs.push(p);
    }
    dirs.sort();

    for d in dirs {
        let p = d.join(".env");
        let Ok(txt) = std::fs::read_to_string(&p) else {
            continue;
        };
        // Cheap pre-scan: only parse if it contains plausible keys.
        if !txt.contains("OPENROUTER_API_KEY")
            && !txt.contains("OPENAI_API_KEY")
            && !txt.contains("GROQ_API_KEY")
        {
            continue;
        }
        for (k, v) in parse_dotenv(&p) {
            if std::env::var(&k).ok().as_deref().unwrap_or("").is_empty() {
                std::env::set_var(k, v);
            }
        }
        if has_any_llm_key() {
            break;
        }
    }
}

pub fn find_lean_repo_root(start: &Path) -> Result<PathBuf, String> {
    fn find_upward(start: &Path) -> Result<PathBuf, String> {
        let mut cur = start
            .canonicalize()
            .map_err(|e| format!("failed to resolve start path {}: {}", start.display(), e))?;
        for _ in 0..80 {
            let has_toolchain = cur.join("lean-toolchain").exists();
            let has_lakefile =
                cur.join("lakefile.lean").exists() || cur.join("lakefile.toml").exists();
            if has_toolchain && has_lakefile {
                return Ok(cur);
            }
            if let Some(parent) = cur.parent() {
                let parent = parent.to_path_buf();
                if parent == cur {
                    break;
                }
                cur = parent;
            } else {
                break;
            }
        }
        Err(format!(
            "Could not find Lean repo root from {} (expected lean-toolchain + lakefile.*)",
            start.display()
        ))
    }

    // Fast path: traditional layout (repo root contains lean-toolchain + lakefile).
    if let Ok(r) = find_upward(start) {
        return Ok(r);
    }

    // Detect Lean 3 projects early, so we don't waste time trying to treat them as Lake repos.
    // (Typical marker: `leanpkg.toml` at repo root.)
    {
        let mut cur = start
            .canonicalize()
            .map_err(|e| format!("failed to resolve start path {}: {}", start.display(), e))?;
        for _ in 0..80 {
            if cur.join("leanpkg.toml").exists() {
                return Err(format!(
                    "Lean 3 project detected at {} (found leanpkg.toml). proofpatch-core currently supports Lean 4 (Lake) projects only.",
                    cur.display()
                ));
            }
            if let Some(parent) = cur.parent() {
                let parent = parent.to_path_buf();
                if parent == cur {
                    break;
                }
                cur = parent;
            } else {
                break;
            }
        }
    }

    // Common alternative: the Lean project lives in a subdirectory (e.g. `lean/`).
    // We keep this heuristic shallow to avoid crawling big trees.
    for child in ["lean", "Lean", "proofs", "formal", "Formal"].iter() {
        let cand = start.join(child);
        if cand.is_dir() {
            if let Ok(r) = find_upward(&cand) {
                return Ok(r);
            }
        }
    }

    find_upward(start)
}

pub fn resolve_lake() -> PathBuf {
    if let Ok(lake_env) = std::env::var("LAKE") {
        let lake_env = lake_env.trim().to_string();
        if !lake_env.is_empty() {
            return PathBuf::from(lake_env);
        }
    }
    if let Some(home) = dirs::home_dir() {
        let elan_lake = home.join(".elan").join("bin").join("lake");
        if elan_lake.exists() {
            return elan_lake;
        }
    }
    PathBuf::from("lake")
}

fn decl_header_regex(decl_name: &str) -> Result<Regex, String> {
    // Many repos use `def`/`abbrev` for exercises and examples, not just `theorem|lemma`.
    // We keep this permissive and anchored, to avoid accidental matches.
    let decl = regex::escape(decl_name);
    let pat = format!(r"^\s*(theorem|lemma|def|abbrev|instance)\s+{}\b", decl);
    Regex::new(&pat).map_err(|e| format!("invalid decl regex: {}", e))
}

pub fn extract_decl_block(text: &str, decl_name: &str) -> Result<String, String> {
    let lines: Vec<&str> = text.lines().collect();
    let pat = decl_header_regex(decl_name)?;

    let start = lines
        .iter()
        .position(|ln| pat.is_match(ln))
        .ok_or_else(|| format!("Could not find theorem/lemma/def named {}", decl_name))?;

    // Try to find `:=` line to anchor the end of the signature; then include some tail context.
    let mut end = None;
    for j in start..usize::min(lines.len(), start + 250) {
        if lines[j].contains(":=") {
            end = Some(j);
            break;
        }
    }
    let end = end.unwrap_or_else(|| usize::min(lines.len().saturating_sub(1), start + 80));
    let tail_end = usize::min(lines.len(), end + 40);
    Ok(lines[start..tail_end].join("\n"))
}

pub fn decl_block_contains_sorry(text: &str, decl_name: &str) -> Result<bool, String> {
    let block = extract_decl_block(text, decl_name)?;
    Ok(Regex::new(r"\b(sorry|admit)\b")
        .map_err(|e| format!("invalid sorry/admit regex: {}", e))?
        .is_match(&block))
}

pub fn patch_first_sorry_in_decl(
    text: &str,
    decl_name: &str,
    replacement: &str,
) -> Result<PatchResult, String> {
    let mut lines: Vec<String> = text.lines().map(|s| s.to_string()).collect();
    let pat = decl_header_regex(decl_name)?;
    let start = lines
        .iter()
        .position(|ln| pat.is_match(ln))
        .ok_or_else(|| format!("Could not find theorem/lemma/def named {}", decl_name))?;

    let stop = usize::min(lines.len(), start + 350);
    let sorry_pat =
        Regex::new(r"\b(sorry|admit)\b").map_err(|e| format!("invalid sorry/admit regex: {}", e))?;

    let mut sorry_line = None;
    for j in start..stop {
        let ln = &lines[j];
        if ln.trim_start().starts_with("--") {
            continue;
        }
        if sorry_pat.is_match(ln) {
            sorry_line = Some(j);
            break;
        }
    }
    let Some(sorry_line) = sorry_line else {
        return Err(format!(
            "Could not find a `sorry`/`admit` token inside {}",
            decl_name
        ));
    };

    let repl = replacement.trim_end_matches('\n');
    if repl.trim().is_empty() {
        return Err("Empty replacement.".to_string());
    }

    let before = lines[sorry_line].clone();
    let ws_indent = before
        .chars()
        .take_while(|c| c.is_whitespace())
        .collect::<String>();

    // Replace the *token* `sorry` in-place when it occurs in a larger line (e.g. `def ... := sorry`,
    // `pure_bind := by sorry`), instead of replacing the entire line.
    let m = sorry_pat
        .find(&before)
        .ok_or_else(|| "internal error: sorry regex matched earlier but not now".to_string())?;
    let prefix = &before[..m.start()];
    let suffix = &before[m.end()..];

    let mut repl_lines: Vec<&str> = repl.lines().collect();
    // Common ergonomic case: the original text already has `... := by sorry`,
    // but callers provide a replacement starting with `by ...`. Avoid `by by ...`.
    let prefix_trim = prefix.trim_end();
    let prefix_has_by = prefix_trim.ends_with("by")
        && (prefix_trim.len() == 2
            || prefix_trim
                .as_bytes()
                .get(prefix_trim.len().saturating_sub(3))
                .is_some_and(|c| c.is_ascii_whitespace()));
    if prefix_has_by && !repl_lines.is_empty() {
        if repl_lines[0].trim() == "by" {
            repl_lines.remove(0);
        } else if repl_lines.len() == 1 {
            let one = repl_lines[0].trim_start();
            if one == "by" {
                repl_lines[0] = "";
            } else if let Some(rest) = one.strip_prefix("by ") {
                repl_lines[0] = rest;
            }
        }
    }
    // If we just dropped a leading `by`, it is common that the next line is indented
    // (because it used to be inside a `by` block). When splicing into `... := by sorry`,
    // we want `by simp`, not `by   simp`.
    if prefix_has_by && !repl_lines.is_empty() {
        repl_lines[0] = repl_lines[0].trim_start();
    }
    if repl_lines.is_empty() || repl_lines.iter().all(|l| l.trim().is_empty()) {
        return Err("Replacement reduced to empty after `by` normalization.".to_string());
    }
    let mut new_lines: Vec<String> = Vec::new();
    if repl_lines.len() == 1 {
        new_lines.push(format!("{}{}{}", prefix, repl_lines[0], suffix));
    } else {
        new_lines.push(format!("{}{}", prefix, repl_lines[0]));
        for mid in &repl_lines[1..repl_lines.len() - 1] {
            if mid.trim().is_empty() {
                new_lines.push(mid.to_string());
            } else {
                new_lines.push(format!("{}{}", ws_indent, mid));
            }
        }
        let last = repl_lines[repl_lines.len() - 1];
        if last.trim().is_empty() {
            new_lines.push(format!("{}{}", last, suffix));
        } else {
            new_lines.push(format!("{}{}{}", ws_indent, last, suffix));
        }
    }

    lines.splice(sorry_line..=sorry_line, new_lines.clone());
    let after = new_lines.join("\n");
    let indent = ws_indent;
    let mut out_text = lines.join("\n");
    if text.ends_with('\n') {
        out_text.push('\n');
    }

    Ok(PatchResult {
        text: out_text,
        changed: true,
        line: sorry_line + 1,
        indent,
        before,
        after,
    })
}

pub fn patch_first_sorry_in_region(
    text: &str,
    start_line_1: usize,
    end_line_1_inclusive: usize,
    replacement: &str,
) -> Result<PatchResult, String> {
    if start_line_1 == 0 {
        return Err("start_line must be >= 1".to_string());
    }
    if end_line_1_inclusive == 0 {
        return Err("end_line must be >= 1".to_string());
    }
    if end_line_1_inclusive < start_line_1 {
        return Err("end_line must be >= start_line".to_string());
    }

    let mut lines: Vec<String> = text.lines().map(|s| s.to_string()).collect();
    if lines.is_empty() {
        return Err("Empty file".to_string());
    }

    let start0 = start_line_1 - 1;
    let end0 = usize::min(end_line_1_inclusive - 1, lines.len() - 1);
    if start0 >= lines.len() {
        return Err(format!(
            "start_line {} out of range (file has {} lines)",
            start_line_1,
            lines.len()
        ));
    }

    let repl = replacement.trim_end_matches('\n');
    if repl.trim().is_empty() {
        return Err("Empty replacement.".to_string());
    }

    let sorry_pat =
        Regex::new(r"\b(sorry|admit)\b").map_err(|e| format!("invalid sorry/admit regex: {}", e))?;
    let mut sorry_line = None;
    for j in start0..=end0 {
        let ln = &lines[j];
        if ln.trim_start().starts_with("--") {
            continue;
        }
        if sorry_pat.is_match(ln) {
            sorry_line = Some(j);
            break;
        }
    }

    let Some(sorry_line) = sorry_line else {
        return Err(format!(
            "Could not find a `sorry`/`admit` token between lines {}..={}",
            start_line_1, end_line_1_inclusive
        ));
    };

    let before = lines[sorry_line].clone();
    let ws_indent = before
        .chars()
        .take_while(|c| c.is_whitespace())
        .collect::<String>();

    let m = sorry_pat
        .find(&before)
        .ok_or_else(|| "internal error: sorry regex matched earlier but not now".to_string())?;
    let prefix = &before[..m.start()];
    let suffix = &before[m.end()..];

    let mut repl_lines: Vec<&str> = repl.lines().collect();
    let prefix_trim = prefix.trim_end();
    let prefix_has_by = prefix_trim.ends_with("by")
        && (prefix_trim.len() == 2
            || prefix_trim
                .as_bytes()
                .get(prefix_trim.len().saturating_sub(3))
                .is_some_and(|c| c.is_ascii_whitespace()));
    if prefix_has_by && !repl_lines.is_empty() {
        if repl_lines[0].trim() == "by" {
            repl_lines.remove(0);
        } else if repl_lines.len() == 1 {
            let one = repl_lines[0].trim_start();
            if one == "by" {
                repl_lines[0] = "";
            } else if let Some(rest) = one.strip_prefix("by ") {
                repl_lines[0] = rest;
            }
        }
    }
    if prefix_has_by && !repl_lines.is_empty() {
        repl_lines[0] = repl_lines[0].trim_start();
    }
    if repl_lines.is_empty() || repl_lines.iter().all(|l| l.trim().is_empty()) {
        return Err("Replacement reduced to empty after `by` normalization.".to_string());
    }
    let mut new_lines: Vec<String> = Vec::new();
    if repl_lines.len() == 1 {
        new_lines.push(format!("{}{}{}", prefix, repl_lines[0], suffix));
    } else {
        new_lines.push(format!("{}{}", prefix, repl_lines[0]));
        for mid in &repl_lines[1..repl_lines.len() - 1] {
            if mid.trim().is_empty() {
                new_lines.push(mid.to_string());
            } else {
                new_lines.push(format!("{}{}", ws_indent, mid));
            }
        }
        let last = repl_lines[repl_lines.len() - 1];
        if last.trim().is_empty() {
            new_lines.push(format!("{}{}", last, suffix));
        } else {
            new_lines.push(format!("{}{}{}", ws_indent, last, suffix));
        }
    }

    lines.splice(sorry_line..=sorry_line, new_lines.clone());
    let after = new_lines.join("\n");
    let indent = ws_indent;

    let mut out_text = lines.join("\n");
    if text.ends_with('\n') {
        out_text.push('\n');
    }

    Ok(PatchResult {
        text: out_text,
        changed: true,
        line: sorry_line + 1,
        indent,
        before,
        after,
    })
}

pub fn locate_sorries_in_text(
    text: &str,
    max_results: usize,
    context_lines: usize,
) -> Result<Vec<SorryLocation>, String> {
    let max_results = max_results.max(1).min(500);
    let context_lines = context_lines.min(50);
    let sorry_pat =
        Regex::new(r"\b(sorry|admit)\b").map_err(|e| format!("invalid sorry/admit regex: {}", e))?;

    fn is_ident_char(c: char) -> bool {
        c.is_ascii_alphanumeric() || c == '_' || c == '\''
    }

    fn parse_decl_header(line: &str) -> Option<(&'static str, String)> {
        let t = line.trim_start();
        // Best-effort: keep this conservative; it’s a hint, not a parser.
        for kw in [
            "theorem", "lemma", "def", "abbrev", "instance", "structure", "class",
        ] {
            let prefix = format!("{kw} ");
            if let Some(rest) = t.strip_prefix(&prefix) {
                let rest = rest.trim_start();
                if rest.is_empty() {
                    return None;
                }
                // name is the first token, stopping at whitespace or common delimiters
                let mut name = String::new();
                for ch in rest.chars() {
                    if ch.is_whitespace() || matches!(ch, ':' | '(' | '{') {
                        break;
                    }
                    name.push(ch);
                }
                if name.is_empty() {
                    return None;
                }
                return Some((kw, name));
            }
        }
        None
    }

    fn nearest_decl(
        lines: &[&str],
        line0: usize,
    ) -> (Option<String>, Option<String>, Option<usize>) {
        let mut i = usize::min(line0, lines.len().saturating_sub(1));
        loop {
            if let Some((kw, name)) = parse_decl_header(lines[i]) {
                return (Some(kw.to_string()), Some(name), Some(i + 1));
            }
            if i == 0 {
                break;
            }
            i -= 1;
        }
        (None, None, None)
    }

    // Find `sorry`/`admit` occurrences that are not inside:
    // - line comments (`-- ...`)
    // - block comments (`/- ... -/`) [best-effort]
    // - string literals (`"..."`) [best-effort: handles escapes]
    //
    // This is intentionally a lightweight lexer rather than a full Lean parser.
    fn scan_line_for_sorrylike_and_update_state(
        line: &str,
        block_depth_in: usize,
    ) -> (Option<(usize, &'static str)>, usize) {
        let bs = line.as_bytes();
        let mut i = 0usize;
        let mut block_depth = block_depth_in;
        let mut in_string = false;
        let mut escaped = false;

        while i < bs.len() {
            if in_string {
                let b = bs[i];
                if escaped {
                    escaped = false;
                    i += 1;
                    continue;
                }
                if b == b'\\' {
                    escaped = true;
                    i += 1;
                    continue;
                }
                if b == b'"' {
                    in_string = false;
                    i += 1;
                    continue;
                }
                i += 1;
                continue;
            }

            if block_depth > 0 {
                if i + 1 < bs.len() && bs[i] == b'/' && bs[i + 1] == b'-' {
                    block_depth += 1;
                    i += 2;
                    continue;
                }
                if i + 1 < bs.len() && bs[i] == b'-' && bs[i + 1] == b'/' {
                    block_depth = block_depth.saturating_sub(1);
                    i += 2;
                    continue;
                }
                i += 1;
                continue;
            }

            // line comment begins
            if i + 1 < bs.len() && bs[i] == b'-' && bs[i + 1] == b'-' {
                break;
            }
            // block comment begins
            if i + 1 < bs.len() && bs[i] == b'/' && bs[i + 1] == b'-' {
                block_depth = 1;
                i += 2;
                continue;
            }
            // string begins
            if bs[i] == b'"' {
                in_string = true;
                i += 1;
                continue;
            }

            // token match: `sorry` or `admit`
            let token: Option<(&'static [u8], &'static str)> = if i + 4 < bs.len()
                && bs[i] == b's'
                && bs[i + 1] == b'o'
                && bs[i + 2] == b'r'
                && bs[i + 3] == b'r'
                && bs[i + 4] == b'y'
            {
                Some((&b"sorry"[..], "sorry"))
            } else if i + 4 < bs.len()
                && bs[i] == b'a'
                && bs[i + 1] == b'd'
                && bs[i + 2] == b'm'
                && bs[i + 3] == b'i'
                && bs[i + 4] == b't'
            {
                Some((&b"admit"[..], "admit"))
            } else {
                None
            };
            if let Some((tok_bytes, tok_str)) = token {
                let tok_len = tok_bytes.len();
                // word boundary checks (approx; ASCII identifiers only)
                let prev = if i == 0 {
                    None
                } else {
                    line[..i].chars().next_back()
                };
                let next = if i + tok_len >= bs.len() {
                    None
                } else {
                    line[i + tok_len..].chars().next()
                };
                let left_ok = prev.map(|c| !is_ident_char(c)).unwrap_or(true);
                let right_ok = next.map(|c| !is_ident_char(c)).unwrap_or(true);
                if left_ok && right_ok {
                    return (Some((i, tok_str)), block_depth);
                }
            }

            i += 1;
        }

        (None, block_depth)
    }

    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Ok(vec![]);
    }

    let mut out = Vec::new();
    let mut block_depth = 0usize;
    for (i0, ln) in lines.iter().enumerate() {
        // Fast skip for comment-only lines.
        if ln.trim_start().starts_with("--") {
            continue;
        }

        // Prefer token-aware scanning to avoid `"sorry"`/`"admit"` in strings / comments.
        let (hit_opt, new_block_depth) = scan_line_for_sorrylike_and_update_state(ln, block_depth);
        block_depth = new_block_depth;
        let Some((byte_pos, token)) = hit_opt else {
            continue;
        };

        // Confirm regex sees the same match; column uses the byte position.
        let Some(m) = sorry_pat.find(ln) else {
            continue;
        };
        if m.start() != byte_pos {
            // The regex match may have hit a different occurrence (e.g. string vs code).
            // Use the code-based byte_pos for the report.
        }

        {
            let line_1 = i0 + 1;
            let col_1 = byte_pos + 1;

            let region_start = line_1.saturating_sub(2).max(1);
            let region_end = (line_1 + 2).min(lines.len());

            let excerpt_start0 = region_start.saturating_sub(1).saturating_sub(context_lines);
            let excerpt_end0 = (region_end - 1 + context_lines).min(lines.len().saturating_sub(1));
            let excerpt = lines[excerpt_start0..=excerpt_end0].join("\n");
            let (decl_kind, decl_name, decl_line) = nearest_decl(&lines, i0);

            out.push(SorryLocation {
                token: token.to_string(),
                decl_kind,
                decl_name,
                decl_line,
                line: line_1,
                col: col_1,
                line_text: (*ln).to_string(),
                region_start,
                region_end,
                excerpt,
            });
            if out.len() >= max_results {
                break;
            }
        }
    }
    Ok(out)
}

/// Conservative `sorry`/`admit` token count.
///
/// Unlike `locate_sorries_in_text`, this intentionally counts tokens even inside comments/strings.
/// It is meant for status reporting, not patch-site discovery.
pub fn count_sorry_tokens_conservative(text: &str) -> Result<usize, String> {
    let pat =
        Regex::new(r"\b(sorry|admit)\b").map_err(|e| format!("invalid sorry/admit regex: {e}"))?;
    Ok(pat.find_iter(text).count())
}

pub fn count_sorry_tokens_conservative_in_file(
    repo_root: &Path,
    file_rel: &str,
) -> Result<usize, String> {
    let p = repo_root.join(file_rel);
    if !p.exists() {
        return Err(format!("File not found: {}", p.display()));
    }
    let text = std::fs::read_to_string(&p)
        .map_err(|e| format!("failed to read {}: {}", p.display(), e))?;
    count_sorry_tokens_conservative(&text)
}

pub fn locate_sorries_in_file(
    repo_root: &Path,
    file_rel: &str,
    max_results: usize,
    context_lines: usize,
) -> Result<Vec<SorryLocation>, String> {
    let p = repo_root.join(file_rel);
    if !p.exists() {
        return Err(format!("File not found: {}", p.display()));
    }
    let text = std::fs::read_to_string(&p)
        .map_err(|e| format!("failed to read {}: {}", p.display(), e))?;
    locate_sorries_in_text(&text, max_results, context_lines)
}

/// Choose a primary `sorry` to work on next.
///
/// Heuristic:
/// - if we have a first error line, choose the `sorry` whose `line` is closest to it
/// - otherwise pick the first `sorry` in source order
pub fn select_primary_sorry(
    first_error_line_1: Option<usize>,
    locs: &[SorryLocation],
) -> Option<SorryLocation> {
    if locs.is_empty() {
        return None;
    }
    if let Some(err_line) = first_error_line_1 {
        locs.iter()
            .min_by_key(|s| (s.line as i64 - err_line as i64).abs())
            .cloned()
    } else {
        locs.first().cloned()
    }
}

pub fn build_proof_prompt(
    repo_root: &Path,
    file_rel: &str,
    decl: &str,
) -> Result<PromptPayload, String> {
    let repo_root = find_lean_repo_root(repo_root)?;
    load_dotenv_smart(&repo_root);

    let p = repo_root.join(file_rel);
    if !p.exists() {
        return Err(format!("File not found: {}", p.display()));
    }
    let txt = std::fs::read_to_string(&p)
        .map_err(|e| format!("failed to read {}: {}", p.display(), e))?;
    let excerpt = extract_decl_block(&txt, decl)?;

    let system = proof_system_prompt();
    let user = proof_user_prompt(&excerpt);

    Ok(PromptPayload {
        repo_root: repo_root.display().to_string(),
        file: p.display().to_string(),
        decl: decl.to_string(),
        excerpt,
        system,
        user,
    })
}

/// System prompt for proof suggestion (reused across CLI/MCP).
///
/// Public invariant: this prompt is meant to be stable-ish, since it affects tool behavior.
pub fn proof_system_prompt() -> String {
    [
        "You are a Lean 4 proof assistant.",
        "Return ONLY Lean code that replaces a single `sorry` (or `admit`) inside a `by` block.",
        "No markdown fences. No commentary. No surrounding `theorem`/`lemma` header.",
        "Prefer short tactic proofs (`simp`, `aesop`, `nlinarith`, `omega`, `ring_nf`) and existing lemmas.",
        "If you cannot complete the proof, return a minimal partial proof with the smallest remaining goal(s).",
    ]
    .join("\n")
}

/// User prompt for proof suggestion, given an excerpt (reused across CLI/MCP).
pub fn proof_user_prompt(excerpt: &str) -> String {
    format!(
        "We are working in a Lean 4 + Mathlib project.\n\
Here is the declaration context (excerpt):\n\n\
{excerpt}\n\n\
Task: provide the Lean proof code that replaces the `sorry`/`admit` (the proof term only)."
    )
}

pub fn build_rubberduck_prompt(
    repo_root: &Path,
    file_rel: &str,
    decl: &str,
    diagnostics: Option<&str>,
) -> Result<PromptPayload, String> {
    let repo_root = find_lean_repo_root(repo_root)?;
    load_dotenv_smart(&repo_root);
    let p = repo_root.join(file_rel);
    if !p.exists() {
        return Err(format!("File not found: {}", p.display()));
    }
    let txt = std::fs::read_to_string(&p)
        .map_err(|e| format!("failed to read {}: {}", p.display(), e))?;
    let excerpt = extract_decl_block(&txt, decl)?;

    let system = [
        "You are a Lean 4 proof assistant helping a human/agent debug and plan.",
        "Do NOT write Lean code.",
        "Return a structured plan for how to eliminate the next `sorry` / error, with small verifiable steps.",
        "Prefer tactics that create tight feedback loops: `simp`, `simp?`, `aesop?`, `exact?`, `apply?`, `have`, `rcases`, `cases`, `by_cases`.",
        "If a goal looks arithmetic/decidable, suggest `omega`, `nlinarith`, `linarith`, `ring_nf`, `norm_num` (only when applicable).",
        "Output format (strict):",
        "1) Key goal-shape guess (1-2 sentences).",
        "2) 3 candidate next steps (each 1-3 lines), including what to inspect in goals/hypotheses.",
        "3) If stuck: 5 lemma-name search keywords (Lean identifiers) likely to exist in mathlib.",
    ]
    .join("\n");

    let mut user = String::new();
    user.push_str("Context: we are editing a Lean 4 + mathlib file.\n");
    if let Some(d) = diagnostics {
        if !d.trim().is_empty() {
            user.push_str("\nRecent Lean diagnostics (raw):\n");
            user.push_str(d.trim());
            user.push('\n');
        }
    }
    user.push_str("\nDeclaration excerpt:\n\n");
    user.push_str(&excerpt);

    // Research hooks for agents (Cursor can execute these via MCP).
    //
    // Note: we keep these as *suggestions*; proofpatch itself does not talk to MCP servers.
    let q = if decl.contains("nathanson") || decl.contains("polygonal") {
        "Fermat polygonal number theorem Nathanson proof b^2 < 4a 3a < b^2 + 2b + 4 Cauchy lemma"
    } else if decl.contains("cauchy_lemma") {
        "Cauchy lemma b^2 < 4a 0 < b^2 + 2b - 3a + 4 a = sum of four squares b = sum of variables"
    } else if decl.contains("sum_three_squares") || decl.contains("Legendre") {
        "sum of three squares theorem residue classes mod 8 remaining cases 2 5 6"
    } else {
        "Lean 4 proof strategy for arithmetic lemmas"
    };
    let web_q = format!("{q} mathlib Lean");
    user.push_str("\n\nResearch (optional):\n");
    user.push_str("- MCP research plan (multi-source):\n");
    // Use JSON so strings are correctly escaped (quotes, backslashes, etc.).
    // Match the CallMcpTool schema: server + toolName + arguments.
    let plan = serde_json::json!({
        "goal": "Collect a reliable statement and a Lean mapping plan.",
        "calls": [
            {
                "server": "user-arxiv-semantic-search-mcp",
                "toolName": "search_papers",
                "arguments": { "query": q },
            },
            {
                "server": "user-tavily-remote-mcp",
                "toolName": "tavily_search",
                "arguments": { "query": web_q, "search_depth": "advanced", "max_results": 5 },
            },
            {
                "server": "user-perplexity",
                "toolName": "search",
                "arguments": {
                    "query": format!("Summarize the key lemma/step for: {q}. Extract variable definitions and constraints, and mention standard references."),
                },
            }
        ],
        "extract": {
            "schema": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "math_statement": { "type": "string" },
                    "variables": { "type": "object", "additionalProperties": { "type": "string" } },
                    "constraints": { "type": "array", "items": { "type": "string" } },
                    "candidate_mathlib_lemmas": { "type": "array", "items": { "type": "string" } },
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": false,
                            "properties": { "title": { "type": "string" }, "url": { "type": "string" } },
                            "required": ["url"]
                        }
                    }
                },
                "required": ["math_statement"]
            }
        }
    });
    user.push_str(&plan.to_string());
    user.push('\n');

    Ok(PromptPayload {
        repo_root: repo_root.display().to_string(),
        file: p.display().to_string(),
        decl: decl.to_string(),
        excerpt,
        system,
        user,
    })
}

pub fn build_rubberduck_prompt_from_excerpt(
    repo_root: &Path,
    file_rel: &str,
    focus_label: &str,
    excerpt: &str,
    diagnostics: Option<&str>,
) -> Result<PromptPayload, String> {
    let repo_root = find_lean_repo_root(repo_root)?;
    load_dotenv_smart(&repo_root);
    let p = repo_root.join(file_rel);
    if !p.exists() {
        return Err(format!("File not found: {}", p.display()));
    }

    let system = [
        "You are a Lean 4 proof assistant helping a human/agent debug and plan.",
        "Do NOT write Lean code.",
        "Return a structured plan for how to eliminate the next `sorry` / error, with small verifiable steps.",
        "Prefer tactics that create tight feedback loops: `simp`, `simp?`, `aesop?`, `exact?`, `apply?`, `have`, `rcases`, `cases`, `by_cases`.",
        "If a goal looks arithmetic/decidable, suggest `omega`, `nlinarith`, `linarith`, `ring_nf`, `norm_num` (only when applicable).",
        "Output format (strict):",
        "1) Key goal-shape guess (1-2 sentences).",
        "2) 3 candidate next steps (each 1-3 lines), including what to inspect in goals/hypotheses.",
        "3) If stuck: 5 lemma-name search keywords (Lean identifiers) likely to exist in mathlib.",
    ]
    .join("\n");

    let mut user = String::new();
    user.push_str("Context: we are editing a Lean 4 + mathlib file.\n");
    if let Some(d) = diagnostics {
        if !d.trim().is_empty() {
            user.push_str("\nRecent Lean diagnostics (raw):\n");
            user.push_str(d.trim());
            user.push('\n');
        }
    }
    user.push_str("\nFocused excerpt:\n\n");
    user.push_str(excerpt);

    // Research hooks for agents (Cursor can execute these via MCP).
    let q = focus_label;
    let web_q = format!("{q} mathlib Lean");
    user.push_str("\n\nResearch (optional):\n");
    user.push_str("- MCP research plan (multi-source):\n");
    let plan = serde_json::json!({
        "goal": "Collect a reliable statement and a Lean mapping plan.",
        "calls": [
            {
                "server": "user-arxiv-semantic-search-mcp",
                "toolName": "search_papers",
                "arguments": { "query": q },
            },
            {
                "server": "user-tavily-remote-mcp",
                "toolName": "tavily_search",
                "arguments": { "query": web_q, "search_depth": "advanced", "max_results": 5 },
            },
            {
                "server": "user-perplexity",
                "toolName": "search",
                "arguments": {
                    "query": format!("Summarize the key lemma/step for: {q}. Extract variable definitions and constraints, and mention standard references."),
                },
            }
        ],
        "extract": {
            "schema": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "math_statement": { "type": "string" },
                    "variables": { "type": "object", "additionalProperties": { "type": "string" } },
                    "constraints": { "type": "array", "items": { "type": "string" } },
                    "candidate_mathlib_lemmas": { "type": "array", "items": { "type": "string" } },
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": false,
                            "properties": { "title": { "type": "string" }, "url": { "type": "string" } },
                            "required": ["url"]
                        }
                    }
                },
                "required": ["math_statement"]
            }
        }
    });
    user.push_str(&plan.to_string());
    user.push('\n');

    Ok(PromptPayload {
        repo_root: repo_root.display().to_string(),
        file: p.display().to_string(),
        decl: focus_label.to_string(),
        excerpt: excerpt.to_string(),
        system,
        user,
    })
}

pub fn build_region_patch_prompt(
    repo_root: &Path,
    file_rel: &str,
    start_line_1: usize,
    end_line_1_inclusive: usize,
    diagnostics: Option<&str>,
) -> Result<PromptPayload, String> {
    let repo_root = find_lean_repo_root(repo_root)?;
    load_dotenv_smart(&repo_root);

    let p = repo_root.join(file_rel);
    if !p.exists() {
        return Err(format!("File not found: {}", p.display()));
    }
    let txt = std::fs::read_to_string(&p)
        .map_err(|e| format!("failed to read {}: {}", p.display(), e))?;
    let lines: Vec<&str> = txt.lines().collect();
    if lines.is_empty() {
        return Err("Empty file".to_string());
    }
    if start_line_1 == 0 || end_line_1_inclusive == 0 {
        return Err("start_line/end_line must be >= 1".to_string());
    }
    if end_line_1_inclusive < start_line_1 {
        return Err("end_line must be >= start_line".to_string());
    }
    if start_line_1 > lines.len() {
        return Err(format!(
            "start_line {} out of range (file has {} lines)",
            start_line_1,
            lines.len()
        ));
    }
    let end0 = usize::min(end_line_1_inclusive, lines.len());
    let excerpt = lines[(start_line_1 - 1)..end0].join("\n");

    let system = [
        "You are a Lean 4 proof assistant.",
        "Return ONLY Lean code that replaces a single `sorry` inside the provided region.",
        "No markdown fences. No commentary.",
        "Do not repeat the surrounding `theorem`/`lemma` header.",
        "Prefer short tactic proofs (`simp`, `aesop`, `nlinarith`, `omega`, `ring_nf`) and existing lemmas.",
        "If you cannot complete the proof, return a minimal partial proof with the smallest remaining goal(s).",
    ]
    .join("\n");

    let mut user = String::new();
    user.push_str("We are working in a Lean 4 + Mathlib project.\n");
    if let Some(d) = diagnostics {
        if !d.trim().is_empty() {
            user.push_str("\nRecent Lean diagnostics (raw):\n");
            user.push_str(d.trim());
            user.push('\n');
        }
    }
    user.push_str(&format!(
        "\nHere is a focused region (lines {}..={}):\n\n",
        start_line_1, end_line_1_inclusive
    ));
    user.push_str(&excerpt);
    user.push_str(
        "\n\nTask: provide the Lean proof code that replaces the `sorry` (the proof term only).",
    );

    Ok(PromptPayload {
        repo_root: repo_root.display().to_string(),
        file: p.display().to_string(),
        decl: format!("region:{}-{}", start_line_1, end_line_1_inclusive),
        excerpt,
        system,
        user,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NearbyDecl {
    /// 1-based line number where the declaration header begins.
    pub line: usize,
    /// The keyword: theorem|lemma|def|abbrev|instance.
    pub kind: String,
    /// The parsed identifier token after the keyword (best-effort).
    pub name: String,
    /// The full header line text.
    pub header: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextFocus {
    /// "decl" or "line"
    pub kind: String,
    /// Optional decl name when kind="decl".
    pub decl: Option<String>,
    /// Optional line when kind="line".
    pub line: Option<usize>,
    /// 1-based excerpt start line.
    pub start_line: usize,
    /// 1-based excerpt end line (inclusive).
    pub end_line: usize,
    /// The excerpt text.
    pub excerpt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPack {
    pub repo_root: String,
    pub file_rel: String,
    pub file_abs: String,
    pub file_lines: usize,
    pub file_bytes: usize,
    pub imports: Vec<String>,
    pub focus: ContextFocus,
    pub nearby_decls: Vec<NearbyDecl>,
}

fn any_decl_header_regex() -> Result<Regex, String> {
    // Lean identifiers can include `.`, `_`, `'`, unicode, etc.
    // We capture the next token up to whitespace/colon/paren as a best-effort name.
    Regex::new(r"^\s*(theorem|lemma|def|abbrev|instance)\s+([^\s:(]+)")
        .map_err(|e| format!("invalid decl-header regex: {}", e))
}

fn extract_decl_span(lines: &[&str], decl_name: &str) -> Result<(usize, usize, String), String> {
    let pat = decl_header_regex(decl_name)?;
    let start0 = lines
        .iter()
        .position(|ln| pat.is_match(ln))
        .ok_or_else(|| format!("Could not find theorem/lemma/def named {}", decl_name))?;

    // Try to find `:=` line to anchor the end of the signature; then include some tail context.
    let mut sig_end0 = None;
    for j0 in start0..usize::min(lines.len(), start0 + 250) {
        if lines[j0].contains(":=") {
            sig_end0 = Some(j0);
            break;
        }
    }
    let sig_end0 =
        sig_end0.unwrap_or_else(|| usize::min(lines.len().saturating_sub(1), start0 + 80));
    let tail_end0_excl = usize::min(lines.len(), sig_end0 + 1 + 40);
    let excerpt = lines[start0..tail_end0_excl].join("\n");
    Ok((start0 + 1, tail_end0_excl, excerpt))
}

fn extract_line_span(
    lines: &[&str],
    line_1: usize,
    context_lines: usize,
) -> (usize, usize, String) {
    let n = lines.len();
    if n == 0 {
        return (1, 1, String::new());
    }
    let line0 = line_1.saturating_sub(1).min(n - 1);
    let start0 = line0.saturating_sub(context_lines);
    let end0 = usize::min(n - 1, line0 + context_lines);
    let excerpt = lines[start0..=end0].join("\n");
    (start0 + 1, end0 + 1, excerpt)
}

fn collect_imports(lines: &[&str], max_imports: usize) -> Vec<String> {
    let mut out = Vec::new();
    for ln in lines.iter().take(200) {
        let t = ln.trim_start();
        if t.starts_with("--") {
            continue;
        }
        if t.starts_with("import ") || t == "import" {
            out.push(t.trim_end().to_string());
            if out.len() >= max_imports {
                break;
            }
            continue;
        }
        // Stop once we leave the import region and hit a non-trivial line.
        if !out.is_empty()
            && !t.trim().is_empty()
            && !t.starts_with("open ")
            && !t.starts_with("set_option")
        {
            break;
        }
    }
    out
}

/// Build an agent-oriented “context pack” for a file + decl/line.
///
/// Goal: give a tool-friendly, JSON-first bundle containing:
/// - imports (for “what modules are in scope”)
/// - a focused excerpt (decl block or a line window)
/// - nearby declaration headers (for local navigation / naming)
pub fn build_context_pack(
    repo_root: &Path,
    file_rel: &str,
    decl: Option<&str>,
    line_1: Option<usize>,
    context_lines: usize,
    nearby_lines: usize,
    max_nearby_decls: usize,
    max_imports: usize,
) -> Result<ContextPack, String> {
    let repo_root = find_lean_repo_root(repo_root)?;
    let p = repo_root.join(file_rel);
    if !p.exists() {
        return Err(format!("File not found: {}", p.display()));
    }

    let txt = std::fs::read_to_string(&p)
        .map_err(|e| format!("failed to read {}: {}", p.display(), e))?;
    let lines: Vec<&str> = txt.lines().collect();
    let file_lines = lines.len();
    let file_bytes = txt.as_bytes().len();

    let imports = collect_imports(&lines, max_imports);

    let focus = if let Some(d) = decl.filter(|s| !s.trim().is_empty()) {
        let (start_line, end_line_excl, excerpt) = extract_decl_span(&lines, d)?;
        ContextFocus {
            kind: "decl".to_string(),
            decl: Some(d.to_string()),
            line: None,
            start_line,
            end_line: end_line_excl,
            excerpt,
        }
    } else if let Some(l1) = line_1 {
        let (start_line, end_line, excerpt) = extract_line_span(&lines, l1, context_lines);
        ContextFocus {
            kind: "line".to_string(),
            decl: None,
            line: Some(l1),
            start_line,
            end_line,
            excerpt,
        }
    } else {
        // Default: top-of-file preamble excerpt.
        let end0 = usize::min(lines.len().saturating_sub(1), 60);
        let excerpt = if lines.is_empty() {
            String::new()
        } else {
            lines[..=end0].join("\n")
        };
        ContextFocus {
            kind: "preamble".to_string(),
            decl: None,
            line: None,
            start_line: 1,
            end_line: end0 + 1,
            excerpt,
        }
    };

    // Nearby decl headers around the focus region.
    let focus_mid = ((focus.start_line + focus.end_line) / 2).max(1);
    let start_near = focus_mid.saturating_sub(nearby_lines).max(1);
    let end_near = usize::min(file_lines.max(1), focus_mid + nearby_lines);
    let decl_pat = any_decl_header_regex()?;

    let mut nearby_decls = Vec::new();
    if !lines.is_empty() {
        for i1 in start_near..=end_near {
            let ln = lines[i1 - 1];
            if let Some(cap) = decl_pat.captures(ln) {
                let kind = cap.get(1).map(|m| m.as_str()).unwrap_or("").to_string();
                let name = cap.get(2).map(|m| m.as_str()).unwrap_or("").to_string();
                nearby_decls.push(NearbyDecl {
                    line: i1,
                    kind,
                    name,
                    header: ln.to_string(),
                });
                if nearby_decls.len() >= max_nearby_decls {
                    break;
                }
            }
        }
    }

    Ok(ContextPack {
        repo_root: repo_root.display().to_string(),
        file_rel: file_rel.to_string(),
        file_abs: p.display().to_string(),
        file_lines,
        file_bytes,
        imports,
        focus,
        nearby_decls,
    })
}

pub async fn verify_lean_text(
    repo_root: &Path,
    lean_text: &str,
    timeout_s: Duration,
) -> Result<VerifyResult, String> {
    let repo_root = find_lean_repo_root(repo_root)?;
    load_dotenv_smart(&repo_root);
    let lake = resolve_lake();

    // NOTE: `needs_process_stdout` is only relevant when the `lsp` feature is enabled.
    // Lean LSP does not surface stdout/stderr logs needed by some oracle-style workflows.
    #[cfg(feature = "lsp")]
    let needs_process_stdout = lean_text.contains("pp_dump")
        || lean_text.contains("ProofpatchTools")
        || lean_text.contains("\"kind\"")
        || lean_text.contains("try simp?")
        || lean_text.contains("try aesop?")
        || lean_text.contains("try apply?")
        || lean_text.contains("try exact?");

    // Fresh repos often need an initial `lake build` so project modules exist as `.olean` in
    // `.lake/build/lib/lean` (Lean's search path is typically olean-based, not source-based).
    // We only do this when the build output directory is missing, and can be disabled.
    //
    // Env:
    // - `PROOFPATCH_AUTO_BUILD`
    let auto_build =
        env_truthy("PROOFPATCH_AUTO_BUILD", true);
    if auto_build && !repo_root.join(".lake/build/lib/lean").exists() {
        let mut build_cmd = Command::new(&lake);
        build_cmd.arg("build").current_dir(&repo_root);
        let build_out = tokio::time::timeout(timeout_s, build_cmd.output())
            .await
            .map_err(|_| "timeout during `lake build`".to_string());
        match build_out {
            Ok(Ok(output)) if output.status.success() => {}
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                return Ok(VerifyResult {
                    ok: false,
                    timeout: false,
                    returncode: output.status.code(),
                    stdout,
                    stderr: format!("`lake build` failed\n{stderr}"),
                    cmd: vec![lake.display().to_string(), "build".to_string()],
                    cwd: repo_root.display().to_string(),
                    tmp_file: None,
                });
            }
            Ok(Err(e)) => {
                return Ok(VerifyResult {
                    ok: false,
                    timeout: false,
                    returncode: None,
                    stdout: String::new(),
                    stderr: format!("failed to run `lake build`: {e}"),
                    cmd: vec![lake.display().to_string(), "build".to_string()],
                    cwd: repo_root.display().to_string(),
                    tmp_file: None,
                });
            }
            Err(_) => {
                return Ok(VerifyResult {
                    ok: false,
                    timeout: true,
                    returncode: None,
                    stdout: String::new(),
                    stderr: "timeout during `lake build`".to_string(),
                    cmd: vec![lake.display().to_string(), "build".to_string()],
                    cwd: repo_root.display().to_string(),
                    tmp_file: None,
                });
            }
        }
    }

    let mut tmp = tempfile::Builder::new()
        .prefix("proofpatch_")
        .suffix(".lean")
        .tempfile()
        .map_err(|e| format!("failed to create temp file: {}", e))?;
    std::io::Write::write_all(&mut tmp, lean_text.as_bytes())
        .map_err(|e| format!("failed to write temp lean file: {}", e))?;
    let tmp_path = tmp.into_temp_path();
    let tmp_path_buf = tmp_path.to_path_buf();

    // Verifier backend:
    // - default: "auto" (try lean-env, fallback to lake env)
    // - override: PROOFPATCH_VERIFY_BACKEND=lake|lean|lsp|auto
    let backend = std::env::var("PROOFPATCH_VERIFY_BACKEND")
        .unwrap_or_else(|_| "auto".to_string())
        .trim()
        .to_lowercase();

    let lean_args = vec![tmp_path_buf.display().to_string()];
    let lake_cmd_vec = vec![
        lake.display().to_string(),
        "env".to_string(),
        "lean".to_string(),
        tmp_path_buf.display().to_string(),
    ];
    let lean_cmd_vec = vec!["lean".to_string(), tmp_path_buf.display().to_string()];

    let (mut ok, mut timeout, mut returncode, mut stdout, mut stderr, cmd_vec) = match backend.as_str() {
        #[cfg(feature = "lsp")]
        "lsp" => {
            // For LSP, prefer a stable path inside repo_root so rootUri contains the file.
            let cache_root = repo_root.join(".generated").join("proofpatch-lsp");
            let _ = std::fs::create_dir_all(&cache_root);
            // Stable path so the LSP server can reuse caches via didChange.
            let p = cache_root.join("proofpatch_buffer.lean");
            let txt = lean_text.to_string();
            match crate::lsp_client::check_text_via_lsp(&repo_root, &p, txt, timeout_s).await {
                Ok(diag) => {
                    // By default, avoid falling back to spawning `lean`/`lake env lean` just to recover
                    // oracle-style output. That fallback can be extremely slow on some repos/files.
                    //
                    // If you *must* force process-based oracle runs (to recover `Try this:` output, etc),
                    // set:
                    // - PROOFPATCH_ORACLE_FORCE_PROCESS=1
                    let force_process_oracle = env_truthy("PROOFPATCH_ORACLE_FORCE_PROCESS", false);
                    let has_oracle_like_output = (!diag.log_lines.is_empty())
                        || diag
                            .lean_lines
                            .iter()
                            .any(|l| l.contains("\"tool\"") && l.contains("proofpatch"));

                    if needs_process_stdout && force_process_oracle && !has_oracle_like_output {
                        // LSP didn't surface the oracle-style output we need; fall back to process verifier.
                        match run_lean_with_env(&repo_root, &lean_args, timeout_s).await {
                            Ok(r) => (r.0, r.1, r.2, r.3, r.4, lean_cmd_vec),
                            Err(_) => {
                                // Fall back to `lake env lean` so we still capture stdout/stderr.
                                let mut cmd = Command::new(&lake);
                                cmd.arg("env").arg("lean").arg(&tmp_path_buf).current_dir(&repo_root);
                                maybe_extend_lean_path_for_lake_env(&mut cmd);
                                let out = tokio::time::timeout(timeout_s, cmd.output())
                                    .await
                                    .map_err(|_| "timeout".to_string());
                                let r = match out {
                                    Err(_) => (false, true, None, String::new(), String::new()),
                                    Ok(Err(e)) => (
                                        false,
                                        false,
                                        None,
                                        String::new(),
                                        format!("failed to execute: {}", e),
                                    ),
                                    Ok(Ok(output)) => {
                                        let ok = output.status.success();
                                        let returncode = output.status.code();
                                        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                                        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                                        (ok, false, returncode, stdout, stderr)
                                    }
                                };
                                (r.0, r.1, r.2, r.3, r.4, lake_cmd_vec)
                            }
                        }
                    } else {
                        // Synthesize Lean-ish diagnostic lines so existing parsers/counters work.
                        let mut synth = String::new();
                        if !diag.lean_lines.is_empty() {
                            synth = diag.lean_lines.join("\n");
                            synth.push('\n');
                        }
                        // If caller needs stdout-like logs (e.g. `pp_dump` JSON), append captured log lines.
                        if needs_process_stdout && !diag.log_lines.is_empty() {
                            if !synth.ends_with('\n') {
                                synth.push('\n');
                            }
                            synth.push_str(&diag.log_lines.join("\n"));
                            synth.push('\n');
                        }
                        // Keep the buffer file around; it lives under `.generated/`.
                        (
                            diag.ok,
                            false,
                            None,
                            synth,
                            diag.stderr,
                            vec!["lean".to_string(), "--server".to_string()],
                        )
                    }
                }
                Err(e) => (false, true, None, String::new(), format!("lsp backend failed: {e}"), vec!["lean".to_string(), "--server".to_string()]),
            }
        }
        "lake" => {
            let mut cmd = Command::new(&lake);
            cmd.arg("env").arg("lean").arg(&tmp_path_buf).current_dir(&repo_root);
            maybe_extend_lean_path_for_lake_env(&mut cmd);
            let out = tokio::time::timeout(timeout_s, cmd.output())
                .await
                .map_err(|_| "timeout".to_string());
            let r = match out {
                Err(_) => (false, true, None, String::new(), String::new()),
                Ok(Err(e)) => (
                    false,
                    false,
                    None,
                    String::new(),
                    format!("failed to execute: {}", e),
                ),
                Ok(Ok(output)) => {
                    let ok = output.status.success();
                    let returncode = output.status.code();
                    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                    (ok, false, returncode, stdout, stderr)
                }
            };
            (r.0, r.1, r.2, r.3, r.4, lake_cmd_vec)
        }
        "lean" => match run_lean_with_env(&repo_root, &lean_args, timeout_s).await {
            Ok(r) => (r.0, r.1, r.2, r.3, r.4, lean_cmd_vec),
            Err(e) => (false, false, None, String::new(), format!("lean-env backend failed: {e}"), lean_cmd_vec),
        },
        _ => {
            // auto: try lean-env first, then fallback to lake env on failure
            #[cfg(feature = "lsp")]
            {
                // If LSP is compiled in, try it first in auto mode; fall back on failure.
                let cache_root = repo_root.join(".generated").join("proofpatch-lsp");
                let _ = std::fs::create_dir_all(&cache_root);
                let p = cache_root.join("proofpatch_buffer.lean");
                let txt = lean_text.to_string();
                if let Ok(diag) = crate::lsp_client::check_text_via_lsp(&repo_root, &p, txt, timeout_s).await {
                    let mut synth = String::new();
                    if !diag.lean_lines.is_empty() {
                        synth = diag.lean_lines.join("\n");
                        synth.push('\n');
                    }
                    // Keep the buffer file around; it lives under `.generated/`.
                    (diag.ok, false, None, synth, diag.stderr, vec!["lean".to_string(), "--server".to_string()])
                } else {
                    match run_lean_with_env(&repo_root, &lean_args, timeout_s).await {
                        Ok(r) => (r.0, r.1, r.2, r.3, r.4, lean_cmd_vec),
                        Err(_) => {
                            let mut cmd = Command::new(&lake);
                            cmd.arg("env").arg("lean").arg(&tmp_path_buf).current_dir(&repo_root);
                            maybe_extend_lean_path_for_lake_env(&mut cmd);
                            let out = tokio::time::timeout(timeout_s, cmd.output())
                                .await
                                .map_err(|_| "timeout".to_string());
                            let r = match out {
                                Err(_) => (false, true, None, String::new(), String::new()),
                                Ok(Err(e)) => (
                                    false,
                                    false,
                                    None,
                                    String::new(),
                                    format!("failed to execute: {}", e),
                                ),
                                Ok(Ok(output)) => {
                                    let ok = output.status.success();
                                    let returncode = output.status.code();
                                    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                                    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                                    (ok, false, returncode, stdout, stderr)
                                }
                            };
                            (r.0, r.1, r.2, r.3, r.4, lake_cmd_vec)
                        }
                    }
                }
            }
            #[cfg(not(feature = "lsp"))]
            {
                match run_lean_with_env(&repo_root, &lean_args, timeout_s).await {
                    Ok(r) => (r.0, r.1, r.2, r.3, r.4, lean_cmd_vec),
                    Err(_) => {
                        let mut cmd = Command::new(&lake);
                        cmd.arg("env").arg("lean").arg(&tmp_path_buf).current_dir(&repo_root);
                        maybe_extend_lean_path_for_lake_env(&mut cmd);
                        let out = tokio::time::timeout(timeout_s, cmd.output())
                            .await
                            .map_err(|_| "timeout".to_string());
                        let r = match out {
                            Err(_) => (false, true, None, String::new(), String::new()),
                            Ok(Err(e)) => (
                                false,
                                false,
                                None,
                                String::new(),
                                format!("failed to execute: {}", e),
                            ),
                            Ok(Ok(output)) => {
                                let ok = output.status.success();
                                let returncode = output.status.code();
                                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                                (ok, false, returncode, stdout, stderr)
                            }
                        };
                        (r.0, r.1, r.2, r.3, r.4, lake_cmd_vec)
                    }
                }
            }
        }
    };

    // If we hit a missing-olean error, do a one-time `lake build` and retry once.
    if !ok
        && !timeout
        && env_truthy("PROOFPATCH_AUTO_BUILD", true)
        && looks_like_missing_olean(&stdout, &stderr)
    {
        let mut build_cmd = Command::new(&lake);
        build_cmd.arg("build").current_dir(&repo_root);
        let build_out = tokio::time::timeout(timeout_s, build_cmd.output())
            .await
            .map_err(|_| "timeout during `lake build`".to_string());
        match build_out {
            Ok(Ok(output)) if output.status.success() => {}
            Ok(Ok(output)) => {
                let b_stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let b_stderr = String::from_utf8_lossy(&output.stderr).to_string();
                // Return early with the build failure; retrying `lean` will likely be noise.
                ok = false;
                timeout = false;
                returncode = output.status.code();
                stdout = b_stdout;
                stderr = format!("`lake build` failed\n{b_stderr}");
            }
            Ok(Err(e)) => {
                ok = false;
                timeout = false;
                returncode = None;
                stdout = String::new();
                stderr = format!("failed to run `lake build`: {e}");
            }
            Err(_) => {
                ok = false;
                timeout = true;
                returncode = None;
                stdout = String::new();
                stderr = "timeout during `lake build`".to_string();
            }
        }

        if ok {
            // Only retry if build succeeded.
            let retry = match backend.as_str() {
                "lake" => None,
                "lean" => Some("lean"),
                _ => Some("auto"),
            };
            let _ = retry;
            // Retry using the same selection logic (lean-env if available, else lake).
            let r = match backend.as_str() {
                "lake" => {
                    let mut cmd = Command::new(&lake);
                    cmd.arg("env").arg("lean").arg(&tmp_path_buf).current_dir(&repo_root);
                    maybe_extend_lean_path_for_lake_env(&mut cmd);
                    let out2 = tokio::time::timeout(timeout_s, cmd.output())
                        .await
                        .map_err(|_| "timeout".to_string());
                    match out2 {
                        Err(_) => (false, true, None, String::new(), String::new()),
                        Ok(Err(e)) => (
                            false,
                            false,
                            None,
                            String::new(),
                            format!("failed to execute: {}", e),
                        ),
                        Ok(Ok(output)) => {
                            let ok = output.status.success();
                            let returncode = output.status.code();
                            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                            (ok, false, returncode, stdout, stderr)
                        }
                    }
                }
                _ => match run_lean_with_env(&repo_root, &lean_args, timeout_s).await {
                    Ok(r) => (r.0, r.1, r.2, r.3, r.4),
                    Err(_) => {
                        let mut cmd = Command::new(&lake);
                        cmd.arg("env").arg("lean").arg(&tmp_path_buf).current_dir(&repo_root);
                        maybe_extend_lean_path_for_lake_env(&mut cmd);
                        let out2 = tokio::time::timeout(timeout_s, cmd.output())
                            .await
                            .map_err(|_| "timeout".to_string());
                        match out2 {
                            Err(_) => (false, true, None, String::new(), String::new()),
                            Ok(Err(e)) => (
                                false,
                                false,
                                None,
                                String::new(),
                                format!("failed to execute: {}", e),
                            ),
                            Ok(Ok(output)) => {
                                let ok = output.status.success();
                                let returncode = output.status.code();
                                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                                (ok, false, returncode, stdout, stderr)
                            }
                        }
                    }
                },
            };
            ok = r.0;
            timeout = r.1;
            returncode = r.2;
            stdout = r.3;
            stderr = r.4;
        }
    }

    // Ensure the tempfile gets cleaned up (even on timeout).
    let _ = std::fs::remove_file(&tmp_path_buf);

    Ok(VerifyResult {
        ok,
        timeout,
        returncode,
        stdout,
        stderr,
        cmd: cmd_vec,
        cwd: repo_root.display().to_string(),
        // We delete the temp file before returning; keep output truthful.
        tmp_file: None,
    })
}

pub async fn verify_lean_file(
    repo_root: &Path,
    file_rel: &str,
    timeout_s: Duration,
) -> Result<VerifyResult, String> {
    let repo_root = find_lean_repo_root(repo_root)?;
    load_dotenv_smart(&repo_root);
    let p = repo_root.join(file_rel);
    if !p.exists() {
        return Err(format!("File not found: {}", p.display()));
    }

    // Prefer verifying the real file path. This avoids module-resolution problems for repos
    // that use their own module roots (e.g. `MIL.*`) and haven’t been built yet.
    let lake = resolve_lake();
    let auto_build =
        env_truthy("PROOFPATCH_AUTO_BUILD", true);
    // Only build if output dir is missing (avoid redundant builds).
    if auto_build && !repo_root.join(".lake/build/lib/lean").exists() {
        let mut build_cmd = Command::new(&lake);
        build_cmd.arg("build").current_dir(&repo_root);
        let _ = tokio::time::timeout(timeout_s, build_cmd.output()).await;
        // If build fails, `lake env lean` will still likely fail with a clearer message; keep going.
    }

    // Verifier backend:
    // - default: "auto" (try lean-env, fallback to lake env)
    // - override: PROOFPATCH_VERIFY_BACKEND=lake|lean|lsp|auto
    let backend = std::env::var("PROOFPATCH_VERIFY_BACKEND")
        .unwrap_or_else(|_| "auto".to_string())
        .trim()
        .to_lowercase();

    let lake_cmd_vec = vec![
        lake.display().to_string(),
        "env".to_string(),
        "lean".to_string(),
        p.display().to_string(),
    ];
    let lean_cmd_vec = vec!["lean".to_string(), p.display().to_string()];

    let run_lake = || async {
        let mut cmd = Command::new(&lake);
        cmd.arg("env").arg("lean").arg(&p).current_dir(&repo_root);
        maybe_extend_lean_path_for_lake_env(&mut cmd);
        let out = tokio::time::timeout(timeout_s, cmd.output())
            .await
            .map_err(|_| "timeout".to_string());
        match out {
            Err(_) => (false, true, None, String::new(), String::new()),
            Ok(Err(e)) => (
                false,
                false,
                None,
                String::new(),
                format!("failed to execute: {}", e),
            ),
            Ok(Ok(output)) => {
                let ok = output.status.success();
                let returncode = output.status.code();
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                (ok, false, returncode, stdout, stderr)
            }
        }
    };

    let (ok, timeout, returncode, stdout, stderr, cmd_vec) = match backend.as_str() {
        "lake" => {
            let r = run_lake().await;
            (r.0, r.1, r.2, r.3, r.4, lake_cmd_vec)
        }
        #[cfg(feature = "lsp")]
        "lsp" => {
            let txt = std::fs::read_to_string(&p).unwrap_or_default();
            match crate::lsp_client::check_text_via_lsp(&repo_root, &p, txt, timeout_s).await {
                Ok(diag) => {
                    let mut synth = String::new();
                    if !diag.lean_lines.is_empty() {
                        synth = diag.lean_lines.join("\n");
                        synth.push('\n');
                    }
                    (diag.ok, false, None, synth, diag.stderr, vec!["lean".to_string(), "--server".to_string()])
                }
                Err(_) => {
                    // fall back to lean-env (captured `lake env env`) for file verification
                    match run_lean_with_env(&repo_root, &[p.display().to_string()], timeout_s).await {
                        Ok(r) => (r.0, r.1, r.2, r.3, r.4, lean_cmd_vec),
                        Err(e) => (false, false, None, String::new(), format!("lean-env backend failed: {e}"), lean_cmd_vec),
                    }
                }
            }
        }
        "lean" => match run_lean_with_env(&repo_root, &[p.display().to_string()], timeout_s).await {
            Ok(r) => (r.0, r.1, r.2, r.3, r.4, lean_cmd_vec),
            Err(e) => (false, false, None, String::new(), format!("lean-env backend failed: {e}"), lean_cmd_vec),
        },
        _ => {
            // auto: try lean-env first, then fallback to lake env on failure
            #[cfg(feature = "lsp")]
            {
                let txt = std::fs::read_to_string(&p).unwrap_or_default();
                if let Ok(diag) = crate::lsp_client::check_text_via_lsp(&repo_root, &p, txt, timeout_s).await {
                    let mut synth = String::new();
                    if !diag.lean_lines.is_empty() {
                        synth = diag.lean_lines.join("\n");
                        synth.push('\n');
                    }
                    (diag.ok, false, None, synth, diag.stderr, vec!["lean".to_string(), "--server".to_string()])
                } else {
                    match run_lean_with_env(&repo_root, &[p.display().to_string()], timeout_s).await {
                        Ok(r) => (r.0, r.1, r.2, r.3, r.4, lean_cmd_vec),
                        Err(_) => {
                            let r = run_lake().await;
                            (r.0, r.1, r.2, r.3, r.4, lake_cmd_vec)
                        }
                    }
                }
            }
            #[cfg(not(feature = "lsp"))]
            {
                match run_lean_with_env(&repo_root, &[p.display().to_string()], timeout_s).await {
                    Ok(r) => (r.0, r.1, r.2, r.3, r.4, lean_cmd_vec),
                    Err(_) => {
                        let r = run_lake().await;
                        (r.0, r.1, r.2, r.3, r.4, lake_cmd_vec)
                    }
                }
            }
        }
    };

    // If we hit a missing-olean error, do a one-time `lake build` and retry once.
    let (ok, timeout, returncode, stdout, stderr) = if !ok
        && !timeout
        && env_truthy("PROOFPATCH_AUTO_BUILD", true)
        && looks_like_missing_olean(&stdout, &stderr)
    {
        let mut build_cmd = Command::new(&lake);
        build_cmd.arg("build").current_dir(&repo_root);
        let build_out = tokio::time::timeout(timeout_s, build_cmd.output())
            .await
            .map_err(|_| "timeout during `lake build`".to_string());
        match build_out {
            Ok(Ok(output)) if output.status.success() => {}
            Ok(Ok(output)) => {
                let b_stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let b_stderr = String::from_utf8_lossy(&output.stderr).to_string();
                return Ok(VerifyResult {
                    ok: false,
                    timeout: false,
                    returncode: output.status.code(),
                    stdout: b_stdout,
                    stderr: format!("`lake build` failed\n{b_stderr}"),
                    cmd: vec![lake.display().to_string(), "build".to_string()],
                    cwd: repo_root.display().to_string(),
                    tmp_file: None,
                });
            }
            Ok(Err(e)) => {
                return Ok(VerifyResult {
                    ok: false,
                    timeout: false,
                    returncode: None,
                    stdout: String::new(),
                    stderr: format!("failed to run `lake build`: {e}"),
                    cmd: vec![lake.display().to_string(), "build".to_string()],
                    cwd: repo_root.display().to_string(),
                    tmp_file: None,
                });
            }
            Err(_) => {
                return Ok(VerifyResult {
                    ok: false,
                    timeout: true,
                    returncode: None,
                    stdout: String::new(),
                    stderr: "timeout during `lake build`".to_string(),
                    cmd: vec![lake.display().to_string(), "build".to_string()],
                    cwd: repo_root.display().to_string(),
                    tmp_file: None,
                });
            }
        }

        // Re-run verification after build using the same backend selection logic.
        match backend.as_str() {
            "lake" => run_lake().await,
            _ => match run_lean_with_env(&repo_root, &[p.display().to_string()], timeout_s).await {
                Ok(r) => r,
                Err(_) => run_lake().await,
            },
        }
    } else {
        (ok, timeout, returncode, stdout, stderr)
    };

    Ok(VerifyResult {
        ok,
        timeout,
        returncode,
        stdout,
        stderr,
        cmd: cmd_vec,
        cwd: repo_root.display().to_string(),
        tmp_file: None,
    })
}

fn insert_after_imports(text: &str, insert: &str) -> String {
    let mut lines: Vec<String> = text.lines().map(|s| s.to_string()).collect();
    if lines.is_empty() {
        return insert.to_string();
    }

    let mut insert_at = 0usize;
    for (i, ln) in lines.iter().take(200).enumerate() {
        let t = ln.trim_start();
        if t.starts_with("import ") || t == "import" {
            insert_at = i + 1;
        }
    }

    let insert = insert.trim_matches('\n');
    if !insert.trim().is_empty() {
        let insert_lines: Vec<String> = insert.lines().map(|s| s.to_string()).collect();
        lines.splice(insert_at..insert_at, insert_lines);
    }

    let mut out_text = lines.join("\n");
    if text.ends_with('\n') {
        out_text.push('\n');
    }
    out_text
}

fn extract_json_object_by_brace_balance(s: &str) -> Vec<serde_json::Value> {
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut depth: i64 = 0;
    let mut collecting = false;

    for line in s.lines() {
        if !collecting {
            // Lean prints info/warning logs with prefixes like:
            //   "warning: { ...json... }"
            // so don't require JSON to start at column 0.
            if let Some(idx) = line.find('{') {
                collecting = true;
                buf.clear();
                depth = 0;
                buf.push_str(&line[idx..]);
                buf.push('\n');
                depth += line[idx..].chars().filter(|c| *c == '{').count() as i64;
                depth -= line[idx..].chars().filter(|c| *c == '}').count() as i64;
                if depth == 0 {
                    collecting = false;
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(buf.trim()) {
                        if v.is_object() {
                            out.push(v);
                        }
                    }
                }
                continue;
            } else {
                continue;
            }
        }

        depth += line.chars().filter(|c| *c == '{').count() as i64;
        depth -= line.chars().filter(|c| *c == '}').count() as i64;
        buf.push_str(line);
        buf.push('\n');

        if collecting && depth == 0 {
            collecting = false;
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(buf.trim()) {
                if v.is_object() {
                    out.push(v);
                }
            }
        }
    }

    out
}

/// Derive a small candidate list from a pretty-printed goal.
///
/// This is intentionally heuristic and bounded; callers can always supply their own candidates.
pub fn derive_candidates_from_goal_pretty(goal_pretty: &str) -> Vec<String> {
    let g = goal_pretty;
    let mut out: Vec<String> = Vec::new();

    // Always include a few cheap tries.
    out.push("by\n  simp".to_string());
    out.push("by\n  assumption".to_string());
    out.push("by\n  aesop".to_string());
    out.push("by\n  aesop?".to_string());
    out.push("by\n  simp_all".to_string());

    // Goal-shape heuristics.
    let has_ineq = g.contains("≤") || g.contains("≥") || g.contains("<") || g.contains(">");
    let has_pow = g.contains("^");
    let has_mul = g.contains("*");
    let has_nat = g.contains("Nat") || g.contains("ℕ");
    let has_int = g.contains("Int") || g.contains("ℤ");
    let has_real = g.contains("Real") || g.contains("ℝ");

    if has_ineq && (has_nat || has_int) {
        out.push("by\n  omega".to_string());
    }
    if has_ineq && (has_real || has_mul || has_pow) {
        out.push("by\n  nlinarith".to_string());
        out.push("by\n  linarith".to_string());
    }
    if has_pow || has_mul {
        out.push("by\n  ring_nf".to_string());
        out.push("by\n  norm_num".to_string());
    }

    // Many mathlib goals require classical instances for automation.
    out.push("by\n  classical\n  simp".to_string());
    out.push("by\n  classical\n  assumption".to_string());
    out.push("by\n  classical\n  aesop".to_string());

    // Deduplicate while preserving order.
    let mut seen = std::collections::HashSet::new();
    out.retain(|s| seen.insert(s.clone()));
    out.truncate(12);
    out
}

/// Derive a small candidate list from a goal target + local context snippets (`hyps_texts`).
///
/// This is intentionally heuristic and bounded. It should never assume a specific backend
/// representation (we only use strings).
pub fn derive_candidates_from_goal_context(hyps_texts: &[String], target: &str) -> Vec<String> {
    // Build a compact “goal pretty” surface that includes the target plus a few hypotheses.
    // This lets `derive_candidates_from_goal_pretty` pick up on Nat/Int/Real heuristics.
    let mut s = String::new();
    if !target.trim().is_empty() {
        s.push_str(target.trim());
        s.push('\n');
    }
    for h in hyps_texts.iter().take(24) {
        let t = h.trim();
        if t.is_empty() {
            continue;
        }
        s.push_str(t);
        s.push('\n');
    }
    derive_candidates_from_goal_pretty(&s)
}

/// Create and verify a temporary "goal dump" variant of a file near the primary `sorry`.
///
/// This does NOT modify the repo. It returns:
/// - the selected region
/// - the parsed `pp_dump` JSON object (if found)
/// - a verify summary for the temp run
pub async fn goal_dump_nearest(
    repo_root: &Path,
    file_rel: &str,
    timeout_s: Duration,
) -> Result<serde_json::Value, String> {
    let repo_root = find_lean_repo_root(repo_root)?;
    load_dotenv_smart(&repo_root);

    let p = repo_root.join(file_rel);
    if !p.exists() {
        return Err(format!("File not found: {}", p.display()));
    }
    let original = std::fs::read_to_string(&p)
        .map_err(|e| format!("failed to read {}: {}", p.display(), e))?;

    // First, get a baseline error location (if any) to choose the closest sorry.
    let baseline = verify_lean_file(&repo_root, file_rel, timeout_s).await?;
    let first_error_line_1 = parse_first_error_loc(&baseline.stdout, &baseline.stderr).map(|l| l.line);

    let locs = locate_sorries_in_text(&original, 50, 1)?;
    let selected = select_primary_sorry(first_error_line_1, &locs)
        .ok_or_else(|| "No `sorry`/`admit` tokens found in file.".to_string())?;

    // Replace the selected `sorry` with a `pp_dump` + `sorry` (so we still get goals).
    // If this is already inside a `by` block, inject tactics only.
    let is_tactic_ctx = is_tactic_context_for_sorry(&original, selected.line, &selected.line_text);
    let replacement = if is_tactic_ctx {
        "pp_dump\nsorry"
    } else {
        "by\n  pp_dump\n  sorry"
    };
    let patched = patch_first_sorry_in_region(
        &original,
        selected.region_start,
        selected.region_end,
        replacement,
    )?;

    // Insert helper tactic definition after the import block (imports must be at file start).
    let prelude = r#"
open Lean Meta Elab Tactic

namespace ProofpatchInline

elab "pp_dump" : tactic => do
  let goals ← getGoals
  let mut goalsJson : Array Json := #[]
  let mut i : Nat := 0
  for g in goals do
    let fmt ← liftMetaM (Lean.Meta.ppGoal g)
    -- Version-robust “local context”: parse it from `ppGoal` output.
    -- This avoids depending on Lean Meta APIs that may differ across toolchains.
    let mut hyps : Array Json := #[]
    for ln in fmt.pretty.splitOn "\n" do
      if hyps.size >= 40 then
        break
      let t := ln.trimAscii.toString
      if t.isEmpty then
        continue
      if t.startsWith "⊢" then
        break
      hyps := hyps.push (Json.mkObj [("text", Json.str t)])
    goalsJson := goalsJson.push (Json.mkObj [
      ("id", Json.num i),
      ("pretty", Json.str fmt.pretty),
      ("hyps", Json.arr hyps)
    ])
    i := i + 1
  let out := Json.mkObj [
    ("tool", Json.str "proofpatch"),
    ("kind", Json.str "pp_dump"),
    ("goals", Json.arr goalsJson)
  ]
  -- Use a warning channel so the message is visible in non-interactive `lean` runs.
  -- Prefix with a newline so the JSON starts at column 0 on its own line, making it easy to parse.
  logWarning m!"\n{toString out}"

end ProofpatchInline
"#;
    let injected = insert_after_imports(&patched.text, prelude);

    let verify = verify_lean_text(&repo_root, &injected, timeout_s).await?;
    let raw_v = serde_json::to_value(&verify)
        .map_err(|e| format!("failed to serialize verify result: {e}"))?;
    let summary = {
        // Keep summary compatible with existing `summarize_verify_like_output` shape.
        let v = serde_json::to_value(&verify).map_err(|e| format!("json: {e}"))?;
        // Reuse existing summarizer in mcp-server; here we do minimal, similar fields.
        let stdout = v.get("stdout").and_then(|x| x.as_str()).unwrap_or("");
        let stderr = v.get("stderr").and_then(|x| x.as_str()).unwrap_or("");
        let first_error_loc = parse_first_error_loc(stdout, stderr).and_then(|loc| serde_json::to_value(loc).ok());
        let errors = stdout.matches(": error:").count()
            + stdout.matches(": error(").count()
            + stderr.matches(": error:").count()
            + stderr.matches(": error(").count();
        let warnings = stdout.matches(": warning:").count()
            + stdout.matches(": warning(").count()
            + stderr.matches(": warning:").count()
            + stderr.matches(": warning(").count();
        serde_json::json!({
            "ok": v.get("ok").and_then(|x| x.as_bool()).unwrap_or(false),
            "timeout": v.get("timeout").and_then(|x| x.as_bool()).unwrap_or(false),
            "returncode": v.get("returncode").cloned().unwrap_or(serde_json::Value::Null),
            "counts": { "errors": errors, "warnings": warnings },
            "first_error": stdout
                .lines()
                .find(|l| l.contains(": error:") || l.contains(": error("))
                .or_else(|| stderr.lines().find(|l| l.contains(": error:") || l.contains(": error("))),
            "first_error_loc": first_error_loc,
        })
    };

    let mut pp_dump: Option<serde_json::Value> = None;
    let merged = format!("{}\n{}", verify.stdout, verify.stderr);
    for obj in extract_json_object_by_brace_balance(&merged) {
        let is_pp = obj
            .get("tool")
            .and_then(|v| v.as_str())
            == Some("proofpatch")
            && obj.get("kind").and_then(|v| v.as_str()) == Some("pp_dump");
        if is_pp {
            pp_dump = Some(obj);
            break;
        }
    }

    let selected_v =
        serde_json::to_value(&selected).map_err(|e| format!("failed to serialize sorry: {e}"))?;

    Ok(serde_json::json!({
        "repo_root": repo_root.display().to_string(),
        "file": file_rel,
        "selected_sorry": selected_v,
        "region": { "start_line": selected.region_start, "end_line": selected.region_end },
        "verify": { "summary": summary, "raw": raw_v },
        "pp_dump": pp_dump.unwrap_or(serde_json::Value::Null),
    }))
}

fn extract_try_this_suggestions(text: &str) -> Vec<String> {
    // mathlib suggestion tactics often emit either:
    //   "Try this: <tactic script>"
    // or:
    //   "Try this:"
    //   "  [apply] refine ..."
    //
    // Keep parsing conservative and bounded.
    let mut out: Vec<String> = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0usize;
    while i < lines.len() {
        let t = lines[i].trim_start();
        if let Some(rest) = t.strip_prefix("Try this:") {
            let s = rest.trim();
            if !s.is_empty() {
                out.push(s.to_string());
            } else {
                // Next non-empty line is usually the suggestion.
                let mut j = i + 1;
                while j < lines.len() && lines[j].trim().is_empty() {
                    j += 1;
                }
                if j < lines.len() {
                    let mut cand = lines[j].trim().to_string();
                    // Strip leading "[tag]" prefix.
                    if cand.starts_with('[') {
                        if let Some(k) = cand.find(']') {
                            cand = cand[k + 1..].trim().to_string();
                        }
                    }
                    if !cand.is_empty() {
                        out.push(cand);
                    }
                }
            }
        }
        i += 1;
    }
    // Dedup + cap.
    let mut seen = std::collections::HashSet::new();
    out.retain(|s| seen.insert(s.clone()));
    out.truncate(12);
    out
}

fn is_tactic_context_for_sorry(text: &str, selected_line_1: usize, selected_line_text: &str) -> bool {
    // This is a heuristic; it should be stable and conservative.
    //
    // We treat a `sorry` as a tactic hole if it is indented under a `by` line.
    //
    // Additionally, treat common tactic-combinator lines as tactic context even when there is no
    // nearby `by` line (e.g. `all_goals sorry`).
    let tsel = selected_line_text.trim();
    if tsel.starts_with("all_goals") {
        return true;
    }
    if tsel.starts_with("by ") || tsel == "by" {
        return true;
    }
    let lines: Vec<&str> = text.lines().collect();
    if selected_line_1 == 0 || selected_line_1 > lines.len() {
        return false;
    }
    let line_idx0 = selected_line_1 - 1;
    let indent = selected_line_text
        .chars()
        .take_while(|c| *c == ' ' || *c == '\t')
        .count();

    let start = line_idx0.saturating_sub(80);
    for k in (start..line_idx0).rev() {
        let l = lines[k];
        let t = l.trim();
        if t.is_empty() || t.starts_with("--") {
            continue;
        }
        // Positive signals (note: `:= by` can occur on a *more-indented* continuation line).
        if t == "by"
            || t.contains(":= by")
            || t.contains("=> by")
            || t.starts_with('·')
            || t.starts_with("case ")
        {
            return true;
        }
        // Stop at a declaration boundary at smaller indentation to avoid drifting into previous decls.
        let ind_k = l.chars().take_while(|c| *c == ' ' || *c == '\t').count();
        if ind_k < indent
            && (t.starts_with("lemma ")
                || t.starts_with("theorem ")
                || t.starts_with("def ")
                || t.starts_with("instance ")
                || t.starts_with("structure "))
        {
            return false;
        }
    }
    false
}

/// Ask Lean/mathlib's own suggestion tactics for candidates near the primary `sorry`.
///
/// This is a “cheap local search oracle”:
/// - patch nearest sorry to run `pp_dump` + `try simp?`/`exact?`/`apply?`
/// - parse `Try this:` suggestions from stdout
/// - return suggestions + goal dump in a JSON object
pub async fn lean_suggest_nearest(
    repo_root: &Path,
    file_rel: &str,
    timeout_s: Duration,
) -> Result<serde_json::Value, String> {
    let repo_root = find_lean_repo_root(repo_root)?;
    load_dotenv_smart(&repo_root);

    let p = repo_root.join(file_rel);
    if !p.exists() {
        return Err(format!("File not found: {}", p.display()));
    }
    let original = std::fs::read_to_string(&p)
        .map_err(|e| format!("failed to read {}: {}", p.display(), e))?;

    let baseline = verify_lean_file(&repo_root, file_rel, timeout_s).await?;
    let first_error_line_1 = parse_first_error_loc(&baseline.stdout, &baseline.stderr).map(|l| l.line);

    lean_suggest_in_text_at(
        &repo_root,
        file_rel,
        &original,
        timeout_s,
        /* focus_line_1 */ None,
        first_error_line_1,
    )
    .await
}

/// Ask Lean/mathlib's suggestion tactics at a `sorry` nearest to `focus_line_1` (if provided),
/// using `base_text` as the current file contents.
///
/// This is the core primitive needed for proof-tree search: as the search edits a sandbox copy,
/// we can keep asking Lean for suggestions at the *current* hole, not just at the original file.
pub async fn lean_suggest_in_text_at(
    repo_root: &Path,
    file_rel: &str,
    base_text: &str,
    timeout_s: Duration,
    focus_line_1: Option<usize>,
    first_error_line_1: Option<usize>,
) -> Result<serde_json::Value, String> {
    let repo_root = find_lean_repo_root(repo_root)?;
    load_dotenv_smart(&repo_root);

    let locs = locate_sorries_in_text(base_text, 200, 1)?;
    if locs.is_empty() {
        return Err("No `sorry`/`admit` tokens found in text.".to_string());
    }
    let selected = if let Some(fl) = focus_line_1 {
        locs.iter()
            .min_by_key(|l| (l.line as i64 - fl as i64).abs())
            .cloned()
            .ok_or_else(|| "No `sorry`/`admit` tokens found in text.".to_string())?
    } else {
        select_primary_sorry(first_error_line_1, &locs)
            .ok_or_else(|| "No `sorry`/`admit` tokens found in text.".to_string())?
    };

    let is_tactic_ctx = is_tactic_context_for_sorry(base_text, selected.line, &selected.line_text);
    // Oracle strategy: try a small sequence of suggestion tactics, stopping once we get non-empty
    // `Try this:` output. This avoids paying for multiple expensive tactics in one run.
    //
    // Env override:
    // - PROOFPATCH_ORACLE_PASSES: max number of tactic passes (default 2)
    let max_passes = std::env::var("PROOFPATCH_ORACLE_PASSES")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(2)
        .clamp(1, 4);

    // Optional: allow the caller (or planner) to control oracle tactic ordering.
    //
    // - PROOFPATCH_ORACLE_TACTICS="simp?,exact?,apply?" (comma-separated)
    // - PROOFPATCH_ORACLE_BAN="aesop?" (comma-separated)
    let ban: HashSet<String> = std::env::var("PROOFPATCH_ORACLE_BAN")
        .ok()
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let mut order: Vec<String> = std::env::var("PROOFPATCH_ORACLE_TACTICS")
        .ok()
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if order.is_empty() {
        order = vec!["simp?".into(), "aesop?".into(), "exact?".into(), "apply?".into()];
    }
    // Filter banned and clamp by pass budget.
    let mut try_tactics: Vec<String> = order
        .into_iter()
        .filter(|t| !ban.contains(t))
        .collect();
    if try_tactics.is_empty() {
        try_tactics = vec!["simp?".into()];
    }
    try_tactics.truncate(max_passes);

    let prelude = r#"
open Lean Meta Elab Tactic

namespace ProofpatchInline

elab "pp_dump" : tactic => do
  let goals ← getGoals
  let mut goalsJson : Array Json := #[]
  let mut i : Nat := 0
  for g in goals do
    let fmt ← liftMetaM (Lean.Meta.ppGoal g)
    -- Version-robust “local context”: parse it from `ppGoal` output.
    -- This avoids depending on Lean Meta APIs that may differ across toolchains.
    let mut hyps : Array Json := #[]
    for ln in fmt.pretty.splitOn "\n" do
      if hyps.size >= 40 then
        break
      let t := ln.trimAscii.toString
      if t.isEmpty then
        continue
      if t.startsWith "⊢" then
        break
      hyps := hyps.push (Json.mkObj [("text", Json.str t)])
    goalsJson := goalsJson.push (Json.mkObj [
      ("id", Json.num i),
      ("pretty", Json.str fmt.pretty),
      ("hyps", Json.arr hyps)
    ])
    i := i + 1
  let out := Json.mkObj [
    ("tool", Json.str "proofpatch"),
    ("kind", Json.str "pp_dump"),
    ("goals", Json.arr goalsJson)
  ]
  -- Use a warning channel so the message is visible in non-interactive `lean` runs.
  logWarning m!"{toString out}"

end ProofpatchInline
"#;
    let mut verify: Option<VerifyResult> = None;
    let mut merged: String = String::new();
    let mut suggestions: Vec<String> = Vec::new();
    let mut oracle_retried = false;

    for (pass_i, t) in try_tactics.iter().enumerate() {
        // If this is a tactic hole, we must not insert a `by` (it will be a syntax error).
        let replacement = if is_tactic_ctx {
            format!("pp_dump\ntry {t}\nsorry")
        } else {
            format!("by\n  pp_dump\n  try {t}\n  sorry")
        };
        let patched = patch_first_sorry_in_region(
            base_text,
            selected.region_start,
            selected.region_end,
            &replacement,
        )?;
        let injected = insert_after_imports(&patched.text, prelude);
        let v = verify_lean_text(&repo_root, &injected, timeout_s).await?;
        merged = format!("{}\n{}", v.stdout, v.stderr);
        suggestions = extract_try_this_suggestions(&merged);
        verify = Some(v);

        // If suggestion tactics are missing (`unknown tactic`), retry once with a minimal oracle so we
        // still get a goal snapshot.
        if merged.contains("error: unknown tactic") && merged.contains("?") {
            oracle_retried = true;
            let minimal_replacement = if is_tactic_ctx {
                "pp_dump\nsorry".to_string()
            } else {
                "by\n  pp_dump\n  sorry".to_string()
            };
            let patched2 = patch_first_sorry_in_region(
                base_text,
                selected.region_start,
                selected.region_end,
                &minimal_replacement,
            )?;
            let injected2 = insert_after_imports(&patched2.text, prelude);
            let v2 = verify_lean_text(&repo_root, &injected2, timeout_s).await?;
            merged = format!("{}\n{}", v2.stdout, v2.stderr);
            suggestions = extract_try_this_suggestions(&merged);
            verify = Some(v2);
            break;
        }

        // If we got something useful, stop early.
        if !suggestions.is_empty() {
            break;
        }

        // Heuristic: `aesop?` is often expensive and rarely helpful for large inequality goals.
        // If the current goal target looks like an arithmetic/inequality statement, skip the next pass.
        if *t == "simp?" {
            if let Some(next_t) = try_tactics.get(pass_i + 1) {
                if *next_t == "aesop?" {
                    // Best-effort parse pp_dump from the current run.
                    let mut target_line: Option<String> = None;
                    for obj in extract_json_object_by_brace_balance(&merged) {
                        let is_pp = obj
                            .get("tool")
                            .and_then(|v| v.as_str())
                            == Some("proofpatch")
                            && obj.get("kind").and_then(|v| v.as_str()) == Some("pp_dump");
                        if !is_pp {
                            continue;
                        }
                        if let Some(goals) = obj.get("goals").and_then(|v| v.as_array()) {
                            if let Some(g0) = goals.first().and_then(|v| v.get("pretty")).and_then(|v| v.as_str()) {
                                for ln in g0.lines() {
                                    if let Some(rest) = ln.trim_start().strip_prefix("⊢") {
                                        target_line = Some(rest.trim().to_string());
                                        break;
                                    }
                                }
                            }
                        }
                        break;
                    }

                    if let Some(tgt) = target_line {
                        let is_ineq = tgt.contains('≤') || tgt.contains('≥') || tgt.contains('<') || tgt.contains('>');
                        if is_ineq {
                            break;
                        }
                    }
                }
            }
        }

        // Don't always spend the full pass budget; if the first pass already took a while,
        // prefer returning pp_dump + empty suggestions.
        if pass_i + 1 >= max_passes {
            break;
        }
    }

    let verify = verify.ok_or_else(|| "oracle: no verification attempt performed".to_string())?;
    let raw_v = serde_json::to_value(&verify)
        .map_err(|e| format!("failed to serialize verify result: {e}"))?;

    // Reuse existing goal dump extraction.
    let mut pp_dump: Option<serde_json::Value> = None;
    for obj in extract_json_object_by_brace_balance(&merged) {
        let is_pp = obj
            .get("tool")
            .and_then(|v| v.as_str())
            == Some("proofpatch")
            && obj.get("kind").and_then(|v| v.as_str()) == Some("pp_dump");
        if is_pp {
            pp_dump = Some(obj);
            break;
        }
    }

    let selected_v =
        serde_json::to_value(&selected).map_err(|e| format!("failed to serialize sorry: {e}"))?;

    Ok(serde_json::json!({
        "repo_root": repo_root.display().to_string(),
        "file": file_rel,
        "selected_sorry": selected_v,
        "region": { "start_line": selected.region_start, "end_line": selected.region_end },
        "verify": { "raw": raw_v },
        "pp_dump": pp_dump.unwrap_or(serde_json::Value::Null),
        "suggestions": suggestions,
        "oracle_retry": { "attempted": oracle_retried },
    }))
}

/// Minimal oracle: run only `pp_dump; sorry` at the selected hole.
///
/// Use this when you want to (a) compute a goal/state signature for caching, or
/// (b) pick the "easiest" next goal without spending budget on suggestion tactics.
pub async fn goal_dump_in_text_at(
    repo_root: &Path,
    file_rel: &str,
    base_text: &str,
    timeout_s: Duration,
    focus_line_1: Option<usize>,
    first_error_line_1: Option<usize>,
) -> Result<serde_json::Value, String> {
    let repo_root = find_lean_repo_root(repo_root)?;
    load_dotenv_smart(&repo_root);

    let locs = locate_sorries_in_text(base_text, 200, 1)?;
    if locs.is_empty() {
        return Err("No `sorry`/`admit` tokens found in text.".to_string());
    }
    let selected = if let Some(fl) = focus_line_1 {
        locs.iter()
            .min_by_key(|l| (l.line as i64 - fl as i64).abs())
            .cloned()
            .ok_or_else(|| "No `sorry`/`admit` tokens found in text.".to_string())?
    } else {
        select_primary_sorry(first_error_line_1, &locs)
            .ok_or_else(|| "No `sorry`/`admit` tokens found in text.".to_string())?
    };

    let is_tactic_ctx = is_tactic_context_for_sorry(base_text, selected.line, &selected.line_text);
    let replacement = if is_tactic_ctx {
        "pp_dump\nsorry"
    } else {
        "by\n  pp_dump\n  sorry"
    };
    let patched = patch_first_sorry_in_region(
        base_text,
        selected.region_start,
        selected.region_end,
        replacement,
    )?;

    let prelude = r#"
open Lean Meta Elab Tactic

namespace ProofpatchInline

elab "pp_dump" : tactic => do
  let goals ← getGoals
  let mut goalsJson : Array Json := #[]
  let mut i : Nat := 0
  for g in goals do
    let fmt ← liftMetaM (Lean.Meta.ppGoal g)
    -- Version-robust “local context”: parse it from `ppGoal` output.
    let mut hyps : Array Json := #[]
    for ln in fmt.pretty.splitOn "\n" do
      if hyps.size >= 40 then
        break
      let t := ln.trimAscii.toString
      if t.isEmpty then
        continue
      if t.startsWith "⊢" then
        break
      hyps := hyps.push (Json.mkObj [("text", Json.str t)])
    goalsJson := goalsJson.push (Json.mkObj [
      ("id", Json.num i),
      ("pretty", Json.str fmt.pretty),
      ("hyps", Json.arr hyps)
    ])
    i := i + 1
  let out := Json.mkObj [
    ("tool", Json.str "proofpatch"),
    ("kind", Json.str "pp_dump"),
    ("goals", Json.arr goalsJson)
  ]
  -- Use a warning channel so the message is visible in non-interactive `lean` runs.
  logWarning m!"{toString out}"

end ProofpatchInline
"#;
    let injected = insert_after_imports(&patched.text, prelude);

    let verify = verify_lean_text(&repo_root, &injected, timeout_s).await?;
    let raw_v = serde_json::to_value(&verify)
        .map_err(|e| format!("failed to serialize verify result: {e}"))?;

    let merged = format!("{}\n{}", verify.stdout, verify.stderr);
    let mut pp_dump: Option<serde_json::Value> = None;
    for obj in extract_json_object_by_brace_balance(&merged) {
        let is_pp = obj
            .get("tool")
            .and_then(|v| v.as_str())
            == Some("proofpatch")
            && obj.get("kind").and_then(|v| v.as_str()) == Some("pp_dump");
        if is_pp {
            pp_dump = Some(obj);
            break;
        }
    }

    let selected_v =
        serde_json::to_value(&selected).map_err(|e| format!("failed to serialize sorry: {e}"))?;

    Ok(serde_json::json!({
        "repo_root": repo_root.display().to_string(),
        "file": file_rel,
        "selected_sorry": selected_v,
        "region": { "start_line": selected.region_start, "end_line": selected.region_end },
        "verify": { "raw": raw_v },
        "pp_dump": pp_dump.unwrap_or(serde_json::Value::Null),
        "oracle_mode": "dump_only",
    }))
}
