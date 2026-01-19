use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tempfile::NamedTempFile;
use tokio::process::Command;

pub mod llm;
pub mod review;

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
    // Keep it simple and fast: line-based scan + regex extraction.
    // Anchor at start-of-line to avoid matching inside multi-line diagnostics.
    let pat = Regex::new(r"(?m)^([^:\n]+\.lean):(\d+):(\d+):\s+error:")
        .ok()?;
    let joined = format!("{stdout}\n{stderr}");
    let cap = pat.captures(&joined)?;
    let path = cap.get(1)?.as_str().to_string();
    let line = cap.get(2)?.as_str().parse::<usize>().ok()?;
    let col = cap.get(3)?.as_str().parse::<usize>().ok()?;
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
        let out = "/Users/arc/x/Foo.lean:12:34: error: boom\nmore";
        let loc = parse_first_error_loc(out, "").expect("expected loc");
        assert_eq!(loc.path, "/Users/arc/x/Foo.lean");
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
/// - `PROOFYLOOPS_MCP_JSON_PATH`: override path to mcp.json (useful for tests)
pub fn load_cursor_mcp_env_if_present() {
    let p = std::env::var("PROOFYLOOPS_MCP_JSON_PATH")
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
/// - `PROOFYLOOPS_DOTENV_SEARCH` (default: on): set to 0/false/off to disable
/// - `PROOFYLOOPS_DOTENV_SEARCH_ROOT` (default: repo_root.parent): override search root
pub fn load_dotenv_smart(repo_root: &Path) {
    // Base: repo-local .env
    load_dotenv_if_present(repo_root);

    // Local convenience: Cursor MCP env blocks (never override).
    load_cursor_mcp_env_if_present();

    if has_any_llm_key() {
        return;
    }

    if !env_truthy("PROOFYLOOPS_DOTENV_SEARCH", true) {
        return;
    }

    let search_root = std::env::var("PROOFYLOOPS_DOTENV_SEARCH_ROOT")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root.parent().unwrap_or(repo_root).to_path_buf());

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
                    "Lean 3 project detected at {} (found leanpkg.toml). proofyloops-core currently supports Lean 4 (Lake) projects only.",
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
    Ok(Regex::new(r"\bsorry\b")
        .map_err(|e| format!("invalid sorry regex: {}", e))?
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
    let sorry_pat = Regex::new(r"\bsorry\b").map_err(|e| format!("invalid sorry regex: {}", e))?;

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
            "Could not find a `sorry` token inside {}",
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

    let sorry_pat = Regex::new(r"\bsorry\b").map_err(|e| format!("invalid sorry regex: {}", e))?;
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
            "Could not find a `sorry` token between lines {}..={}",
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
    let sorry_pat = Regex::new(r"\bsorry\b").map_err(|e| format!("invalid sorry regex: {}", e))?;

    fn is_ident_char(c: char) -> bool {
        c.is_ascii_alphanumeric() || c == '_' || c == '\''
    }

    // Find `sorry` occurrences that are not inside:
    // - line comments (`-- ...`)
    // - block comments (`/- ... -/`) [best-effort]
    // - string literals (`"..."`) [best-effort: handles escapes]
    //
    // This is intentionally a lightweight lexer rather than a full Lean parser.
    fn scan_line_for_sorry_and_update_state(
        line: &str,
        block_depth_in: usize,
    ) -> (Option<usize>, usize) {
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

            // token match: `sorry`
            if i + 4 < bs.len()
                && bs[i] == b's'
                && bs[i + 1] == b'o'
                && bs[i + 2] == b'r'
                && bs[i + 3] == b'r'
                && bs[i + 4] == b'y'
            {
                // word boundary checks (approx; ASCII identifiers only)
                let prev = if i == 0 {
                    None
                } else {
                    line[..i].chars().next_back()
                };
                let next = if i + 5 >= bs.len() {
                    None
                } else {
                    line[i + 5..].chars().next()
                };
                let left_ok = prev.map(|c| !is_ident_char(c)).unwrap_or(true);
                let right_ok = next.map(|c| !is_ident_char(c)).unwrap_or(true);
                if left_ok && right_ok {
                    return (Some(i), block_depth);
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

        // Prefer token-aware scanning to avoid `"sorry"` in strings / comments.
        let (byte_pos_opt, new_block_depth) = scan_line_for_sorry_and_update_state(ln, block_depth);
        block_depth = new_block_depth;
        let Some(byte_pos) = byte_pos_opt else {
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

            out.push(SorryLocation {
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
        "Return ONLY Lean code that replaces a single `sorry` inside a `by` block.",
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
Task: provide the Lean proof code that replaces the `sorry` (the proof term only)."
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
    user.push_str("\n\nTask: provide the Lean proof code that replaces the `sorry` (the proof term only).");

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

fn extract_decl_span(
    lines: &[&str],
    decl_name: &str,
) -> Result<(usize, usize, String), String> {
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
    let sig_end0 = sig_end0.unwrap_or_else(|| usize::min(lines.len().saturating_sub(1), start0 + 80));
    let tail_end0_excl = usize::min(lines.len(), sig_end0 + 1 + 40);
    let excerpt = lines[start0..tail_end0_excl].join("\n");
    Ok((start0 + 1, tail_end0_excl, excerpt))
}

fn extract_line_span(lines: &[&str], line_1: usize, context_lines: usize) -> (usize, usize, String) {
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
        if !out.is_empty() && !t.trim().is_empty() && !t.starts_with("open ") && !t.starts_with("set_option") {
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

    let txt = std::fs::read_to_string(&p).map_err(|e| format!("failed to read {}: {}", p.display(), e))?;
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
        let excerpt = if lines.is_empty() { String::new() } else { lines[..=end0].join("\n") };
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

    // Fresh repos often need an initial `lake build` so project modules exist as `.olean` in
    // `.lake/build/lib/lean` (Lean's search path is typically olean-based, not source-based).
    // We only do this when the build output directory is missing, and can be disabled.
    if env_truthy("PROOFYLOOPS_AUTO_BUILD", true)
        && !repo_root.join(".lake/build/lib/lean").exists()
    {
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

    let mut tmp = NamedTempFile::new().map_err(|e| format!("failed to create temp file: {}", e))?;
    std::io::Write::write_all(&mut tmp, lean_text.as_bytes())
        .map_err(|e| format!("failed to write temp lean file: {}", e))?;
    let tmp_path = tmp.into_temp_path();
    let tmp_path_buf = tmp_path.to_path_buf();

    let cmd_vec = vec![
        lake.display().to_string(),
        "env".to_string(),
        "lean".to_string(),
        tmp_path_buf.display().to_string(),
    ];

    let mut cmd = Command::new(&lake);
    cmd.arg("env")
        .arg("lean")
        .arg(&tmp_path_buf)
        .current_dir(&repo_root);

    let out = tokio::time::timeout(timeout_s, cmd.output())
        .await
        .map_err(|_| "timeout".to_string());

    let (mut ok, mut timeout, mut returncode, mut stdout, mut stderr) = match out {
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

    // If we hit a missing-olean error, do a one-time `lake build` and retry once.
    if !ok
        && !timeout
        && env_truthy("PROOFYLOOPS_AUTO_BUILD", true)
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
            // Only retry `lean` if build succeeded.
            let out2 = tokio::time::timeout(timeout_s, cmd.output())
                .await
                .map_err(|_| "timeout".to_string());
            let r = match out2 {
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
        tmp_file: Some(tmp_path_buf.display().to_string()),
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
    if env_truthy("PROOFYLOOPS_AUTO_BUILD", true)
        && !repo_root.join(".lake/build/lib/lean").exists()
    {
        let mut build_cmd = Command::new(&lake);
        build_cmd.arg("build").current_dir(&repo_root);
        let _ = tokio::time::timeout(timeout_s, build_cmd.output()).await;
        // If build fails, `lake env lean` will still likely fail with a clearer message; keep going.
    }

    let cmd_vec = vec![
        lake.display().to_string(),
        "env".to_string(),
        "lean".to_string(),
        p.display().to_string(),
    ];
    let mut cmd = Command::new(&lake);
    cmd.arg("env").arg("lean").arg(&p).current_dir(&repo_root);
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

    // If we hit a missing-olean error, do a one-time `lake build` and retry once.
    let (ok, timeout, returncode, stdout, stderr) = if !ok
        && !timeout
        && env_truthy("PROOFYLOOPS_AUTO_BUILD", true)
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
