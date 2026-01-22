//! proofpatch MCP server (HTTP via axum-mcp).
//!
//! Exposes `proofpatch-core` as MCP tools over HTTP/stdio.
//!
//! Run:
//! ```bash
//! cargo run --quiet -p proofpatch-mcp --bin proofpatch-mcp
//! ```
//!
//! Then:
//! - `curl http://127.0.0.1:8087/health`
//! - `curl http://127.0.0.1:8087/tools/list`
//! - example call:
//! ```bash
//! curl -X POST http://127.0.0.1:8087/tools/call \
//!   -H 'Content-Type: application/json' \
//!   -d '{"name":"proofpatch_prompt","arguments":{"repo_root":"/abs/path/to/lean-repo","file":"Some/File.lean","lemma":"some_theorem"}}'
//! ```
//!
//! Configuration:
//! - `PROOFPATCH_MCP_ADDR` (default: `127.0.0.1:8087`)
//! - `PROOFPATCH_MCP_TOOL_TIMEOUT_S` (default: `180`)

use async_trait::async_trait;
use axum_mcp::{
    extract_integer_opt, extract_string, extract_string_opt, McpServer, ServerConfig, Tool,
};
use proofpatch_core as plc;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::Duration as StdDuration;

// Optional stdio MCP transport (Cursor can spawn without a daemon).
#[cfg(feature = "stdio")]
use rmcp::{
    handler::server::router::tool::ToolRouter as RmcpToolRouter,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content, Implementation, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
    transport::stdio,
    ErrorData as McpError, ServiceExt,
};
#[cfg(feature = "stdio")]
use schemars::JsonSchema;
#[cfg(feature = "stdio")]
use serde::{Deserialize, Serialize};

fn default_proofpatch_root() -> PathBuf {
    // `.../proofpatch/mcp-server` → `.../proofpatch`
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("mcp-server should be nested under proofpatch/")
        .to_path_buf()
}

fn extract_u64_opt(args: &Value, key: &str) -> Result<Option<u64>, String> {
    match extract_integer_opt(args, key) {
        Some(v) if v >= 0 => Ok(Some(v as u64)),
        Some(_) => Err(format!("Argument `{}` must be a non-negative integer", key)),
        None => Ok(None),
    }
}

fn count_substring(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    haystack.match_indices(needle).count()
}

fn first_diagnostic_line(text: &str) -> Option<String> {
    for line in text.lines() {
        // Lean diagnostics generally look like:
        //   path:line:col: error: ...
        //   path:line:col: warning: ...
        //
        // We keep this deliberately heuristic: it's for an *agent loop* preview;
        // the raw stdout/stderr are preserved in the payload.
        if line.contains(": error:")
            || line.contains(": error(")
            || line.starts_with("error:")
            || line.starts_with("error(")
        {
            return Some(line.to_string());
        }
    }
    None
}

fn parse_first_diagnostic_location(line: &str) -> Option<(String, usize, usize, String)> {
    // Parse:
    //   /path/to/file.lean:74:2: error: ...
    //   /path/to/file.lean:74:2: warning: ...
    //
    // Return: (path, line, col, kind)
    let (kind, idx) = if let Some(i) = line.find(": error:").or_else(|| line.find(": error(")) {
        ("error".to_string(), i)
    } else if let Some(i) = line.find(": warning:") {
        ("warning".to_string(), i)
    } else {
        return None;
    };

    let prefix = line[..idx].trim_end();
    let mut it = prefix.rsplitn(3, ':');
    let col_s = it.next()?;
    let line_s = it.next()?;
    let path = it.next()?.to_string();
    let line_no = line_s.trim().parse::<usize>().ok()?;
    let col_no = col_s.trim().parse::<usize>().ok()?;
    Some((path, line_no, col_no, kind))
}

fn summarize_verify_like_output(raw: &Value) -> Value {
    let ok = raw.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
    let timeout = raw
        .get("timeout")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let returncode = raw.get("returncode").cloned().unwrap_or(Value::Null);

    let stdout = raw.get("stdout").and_then(|v| v.as_str()).unwrap_or("");
    let stderr = raw.get("stderr").and_then(|v| v.as_str()).unwrap_or("");

    let sorry_warnings = count_substring(stdout, "declaration uses 'sorry'")
        + count_substring(stderr, "declaration uses 'sorry'")
        + count_substring(stdout, "declaration uses 'admit'")
        + count_substring(stderr, "declaration uses 'admit'");

    let warning_count = count_substring(stdout, ": warning:")
        + count_substring(stdout, ": warning(")
        + count_substring(stderr, ": warning:")
        + count_substring(stderr, ": warning(")
        + count_substring(stderr, "warning:");
    let error_count = count_substring(stdout, ": error:")
        + count_substring(stdout, ": error(")
        + count_substring(stderr, ": error:")
        + count_substring(stderr, ": error(")
        + count_substring(stderr, "error:");

    let first_error = first_diagnostic_line(stdout).or_else(|| first_diagnostic_line(stderr));
    let first_error_loc = first_error
        .as_deref()
        .and_then(parse_first_diagnostic_location)
        .map(
            |(path, line, col, kind)| json!({"path": path, "line": line, "col": col, "kind": kind}),
        );

    json!({
        "ok": ok,
        "timeout": timeout,
        "returncode": returncode,
        "counts": {
            "errors": error_count,
            "warnings": warning_count,
            "sorry_warnings": sorry_warnings
        },
        "first_error": first_error,
        "first_error_loc": first_error_loc,
    })
}

// NOTE: The MCP server used to bridge to a Python CLI.
// It is Rust-native now (proofpatch-core has the provider router), so there is no reason to shell out.

fn proofpatch_root_from_args(args: &Value) -> Result<PathBuf, String> {
    if let Ok(env_root) = std::env::var("PROOFPATCH_ROOT") {
        if !env_root.trim().is_empty() {
            return Ok(PathBuf::from(env_root));
        }
    }
    let root = extract_string_opt(args, "proofpatch_root")
        .unwrap_or_else(|| default_proofpatch_root().to_string_lossy().to_string());
    Ok(PathBuf::from(root))
}

fn repo_root_from_args(args: &Value) -> Result<PathBuf, String> {
    let repo_root = PathBuf::from(extract_string(args, "repo_root")?);
    // Parse `proofpatch_root` to keep schemas honest, but do not use it for anything.
    // Repo-root resolution is independent of helper-root.
    let _ = proofpatch_root_from_args(args)?;
    Ok(repo_root)
}

fn resolve_lean_repo_root(repo_root: PathBuf, file: Option<&str>) -> Result<PathBuf, String> {
    // Primary: honor the user's repo_root if it's already a Lean project root (or a parent of one).
    if let Ok(r) = plc::find_lean_repo_root(&repo_root) {
        return Ok(r);
    }

    // Helpful fallback: if the user passed a super-workspace root but the *file* points into a
    // Lean project, use the file location as the anchor for upward discovery.
    if let Some(f) = file {
        let p = PathBuf::from(f);
        let abs = if p.is_absolute() { p } else { repo_root.join(p) };
        if let Some(parent) = abs.parent() {
            if let Ok(r) = plc::find_lean_repo_root(parent) {
                return Ok(r);
            }
        }
    }

    // Final: return the original error (with the original start path in the message).
    plc::find_lean_repo_root(&repo_root)
}

struct ProofpatchPromptTool;

#[async_trait]
impl Tool for ProofpatchPromptTool {
    fn description(&self) -> &str {
        "Extract the (system,user) prompt + excerpt for a lemma (`proofpatch prompt`)."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "repo_root": { "type": "string", "description": "Lean repo root" },
                "file": { "type": "string", "description": "File path relative to repo root" },
                "lemma": { "type": "string", "description": "Lemma name to extract" },
                "timeout_s": { "type": "integer", "default": 30 },
                "proofpatch_root": { "type": "string", "description": "Path to proofpatch (defaults to sibling of this crate)" }
            },
            "required": ["repo_root", "file", "lemma"]
        })
    }

    async fn call(&self, args: &Value) -> Result<Value, String> {
        let file = extract_string(args, "file")?;
        let lemma = extract_string(args, "lemma")?;
        let repo_root = repo_root_from_args(args)?;
        // `proofpatch_root` exists only for schema compatibility; we don't use it.
        let _ = extract_u64_opt(args, "timeout_s")?.unwrap_or(30);
        let repo_root = resolve_lean_repo_root(repo_root, Some(&file))?;
        let payload = plc::build_proof_prompt(&repo_root, &file, &lemma)?;
        serde_json::to_value(payload).map_err(|e| format!("failed to serialize payload: {}", e))
    }
}

struct ProofpatchVerifyTool;

#[async_trait]
impl Tool for ProofpatchVerifyTool {
    fn description(&self) -> &str {
        "Elaboration-check a file (`proofpatch verify`)."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "repo_root": { "type": "string" },
                "file": { "type": "string", "description": "File path relative to repo root" },
                "timeout_s": { "type": "integer", "default": 120 },
                "proofpatch_root": { "type": "string" }
            },
            "required": ["repo_root", "file"]
        })
    }

    async fn call(&self, args: &Value) -> Result<Value, String> {
        let file = extract_string(args, "file")?;
        let repo_root = repo_root_from_args(args)?;
        let repo_root = resolve_lean_repo_root(repo_root, Some(&file))?;
        let timeout_s = extract_u64_opt(args, "timeout_s")?.unwrap_or(120);
        let raw =
            plc::verify_lean_file(&repo_root, &file, StdDuration::from_secs(timeout_s)).await?;
        serde_json::to_value(raw).map_err(|e| format!("failed to serialize verify result: {}", e))
    }
}

struct ProofpatchVerifySummaryTool;

#[async_trait]
impl Tool for ProofpatchVerifySummaryTool {
    fn description(&self) -> &str {
        "Elaboration-check a file, returning a small summary plus raw output (`proofpatch verify`)."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "repo_root": { "type": "string" },
                "file": { "type": "string", "description": "File path relative to repo root" },
                "timeout_s": { "type": "integer", "default": 120 },
                "proofpatch_root": { "type": "string" }
            },
            "required": ["repo_root", "file"]
        })
    }

    async fn call(&self, args: &Value) -> Result<Value, String> {
        let file = extract_string(args, "file")?;
        let repo_root = repo_root_from_args(args)?;
        let repo_root = resolve_lean_repo_root(repo_root, Some(&file))?;
        let timeout_s = extract_u64_opt(args, "timeout_s")?.unwrap_or(120);
        let raw =
            plc::verify_lean_file(&repo_root, &file, StdDuration::from_secs(timeout_s)).await?;
        let raw_v = serde_json::to_value(raw)
            .map_err(|e| format!("failed to serialize verify result: {}", e))?;
        Ok(json!({"summary": summarize_verify_like_output(&raw_v), "raw": raw_v}))
    }
}

struct ProofpatchSuggestTool;

#[async_trait]
impl Tool for ProofpatchSuggestTool {
    fn description(&self) -> &str {
        "Suggest a proof by running the configured LLM router (`proofpatch suggest`)."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "repo_root": { "type": "string" },
                "file": { "type": "string" },
                "lemma": { "type": "string" },
                "timeout_s": { "type": "integer", "default": 120 },
                "proofpatch_root": { "type": "string" }
            },
            "required": ["repo_root", "file", "lemma"]
        })
    }

    async fn call(&self, args: &Value) -> Result<Value, String> {
        let file = extract_string(args, "file")?;
        let lemma = extract_string(args, "lemma")?;
        let repo_root = repo_root_from_args(args)?;
        let repo_root = resolve_lean_repo_root(repo_root, Some(&file))?;
        let timeout_s = extract_u64_opt(args, "timeout_s")?.unwrap_or(120);
        // `proofpatch_root` exists only for schema compatibility; we don't use it.
        let _ = proofpatch_root_from_args(args)?;

        let payload = plc::build_proof_prompt(&repo_root, &file, &lemma)?;
        let res = plc::llm::chat_completion(
            &payload.system,
            &payload.user,
            StdDuration::from_secs(timeout_s),
        )
        .await?;

        Ok(json!({
            "provider": res.provider,
            "model": res.model,
            "lemma": lemma,
            "file": payload.file,
            "suggestion": res.content,
            "raw": res.raw
        }))
    }
}

struct ProofpatchPatchTool;

#[async_trait]
impl Tool for ProofpatchPatchTool {
    fn description(&self) -> &str {
        "Patch a lemma’s first `sorry` with provided Lean code, then verify (`proofpatch patch`)."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "repo_root": { "type": "string" },
                "file": { "type": "string" },
                "lemma": { "type": "string" },
                "replacement": { "type": "string", "description": "Lean proof-term text to splice in (no markdown fences)" },
                "timeout_s": { "type": "integer", "default": 120 },
                "write": { "type": "boolean", "default": false, "description": "If true, write the patched text back to the file." },
                "include_raw_verify": {
                    "type": "boolean",
                    "default": false,
                    "description": "If true, include full verify raw output (can be large)."
                },
                "proofpatch_root": { "type": "string" }
            },
            "required": ["repo_root", "file", "lemma", "replacement"]
        })
    }

    async fn call(&self, args: &Value) -> Result<Value, String> {
        let repo_root = repo_root_from_args(args)?;
        let file = extract_string(args, "file")?;
        let lemma = extract_string(args, "lemma")?;
        let replacement = extract_string(args, "replacement")?;
        let timeout_s = extract_u64_opt(args, "timeout_s")?.unwrap_or(120);
        let write = args.get("write").and_then(|v| v.as_bool()).unwrap_or(false);
        let include_raw_verify = args
            .get("include_raw_verify")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let repo_root = resolve_lean_repo_root(repo_root, Some(&file))?;
        plc::load_dotenv_smart(&repo_root);

        let p = repo_root.join(&file);
        if !p.exists() {
            return Err(format!("File not found: {}", p.display()));
        }
        let original = std::fs::read_to_string(&p)
            .map_err(|e| format!("failed to read {}: {}", p.display(), e))?;

        let patched = plc::patch_first_sorry_in_decl(&original, &lemma, &replacement)?;
        let still_has_sorry = plc::decl_block_contains_sorry(&patched.text, &lemma)?;
        let mut written_file: Option<String> = None;
        if write {
            std::fs::write(&p, patched.text.as_bytes())
                .map_err(|e| format!("failed to write {}: {}", p.display(), e))?;
            written_file = Some(p.display().to_string());
        }
        let verify = if write {
            plc::verify_lean_file(&repo_root, &file, StdDuration::from_secs(timeout_s)).await?
        } else {
            plc::verify_lean_text(&repo_root, &patched.text, StdDuration::from_secs(timeout_s))
                .await?
        };
        let verify_raw_v = serde_json::to_value(verify)
            .map_err(|e| format!("failed to serialize verify result: {}", e))?;
        let verify_summary = summarize_verify_like_output(&verify_raw_v);

        Ok(json!({
            "file": p.display().to_string(),
            "lemma": lemma,
            "written_file": written_file,
            "patch": {
                "line": patched.line,
                "before": patched.before,
                "after": patched.after,
                "indent": patched.indent,
            },
            "lemma_still_contains_sorry": still_has_sorry,
            "verify": { "summary": verify_summary, "raw": if include_raw_verify { verify_raw_v } else { serde_json::Value::Null } },
        }))
    }
}

struct ProofpatchPatchRegionTool;

#[async_trait]
impl Tool for ProofpatchPatchRegionTool {
    fn description(&self) -> &str {
        "Patch the first `sorry` within a (line-based) region and verify (works for instance fields / local blocks)."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "repo_root": { "type": "string" },
                "file": { "type": "string" },
                "start_line": { "type": "integer", "description": "1-based start line (inclusive)" },
                "end_line": { "type": "integer", "description": "1-based end line (inclusive)" },
                "replacement": { "type": "string", "description": "Lean text to splice in (no markdown fences)" },
                "write": { "type": "boolean", "default": false, "description": "If true, write the patched text back to the file." },
                "include_raw_verify": {
                    "type": "boolean",
                    "default": false,
                    "description": "If true, include full verify raw output (can be large)."
                },
                "timeout_s": { "type": "integer", "default": 120 }
            },
            "required": ["repo_root", "file", "start_line", "end_line", "replacement"]
        })
    }

    async fn call(&self, args: &Value) -> Result<Value, String> {
        let repo_root = repo_root_from_args(args)?;
        let file = extract_string(args, "file")?;
        let start_line = extract_u64_opt(args, "start_line")?
            .ok_or_else(|| "missing start_line".to_string())? as usize;
        let end_line = extract_u64_opt(args, "end_line")?
            .ok_or_else(|| "missing end_line".to_string())? as usize;
        let replacement = extract_string(args, "replacement")?;
        let write = args.get("write").and_then(|v| v.as_bool()).unwrap_or(false);
        let include_raw_verify = args
            .get("include_raw_verify")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let timeout_s = extract_u64_opt(args, "timeout_s")?.unwrap_or(120);

        let repo_root = resolve_lean_repo_root(repo_root, Some(&file))?;
        plc::load_dotenv_smart(&repo_root);

        let p = repo_root.join(&file);
        if !p.exists() {
            return Err(format!("File not found: {}", p.display()));
        }
        let original = std::fs::read_to_string(&p)
            .map_err(|e| format!("failed to read {}: {}", p.display(), e))?;

        let patched =
            plc::patch_first_sorry_in_region(&original, start_line, end_line, &replacement)?;

        // “still contains sorry” is scoped to the patched region (post-patch).
        let region_text = patched
            .text
            .lines()
            .skip(start_line.saturating_sub(1))
            .take(end_line.saturating_sub(start_line).saturating_add(1))
            .collect::<Vec<_>>()
            .join("\n");
        let region_still_contains_sorry = region_text.contains("sorry");

        let mut written_file: Option<String> = None;
        if write {
            std::fs::write(&p, patched.text.as_bytes())
                .map_err(|e| format!("failed to write {}: {}", p.display(), e))?;
            written_file = Some(p.display().to_string());
        }
        let verify = if write {
            plc::verify_lean_file(&repo_root, &file, StdDuration::from_secs(timeout_s)).await?
        } else {
            plc::verify_lean_text(&repo_root, &patched.text, StdDuration::from_secs(timeout_s))
                .await?
        };
        let verify_raw_v = serde_json::to_value(verify)
            .map_err(|e| format!("failed to serialize verify result: {}", e))?;
        let verify_summary = summarize_verify_like_output(&verify_raw_v);

        Ok(json!({
            "file": p.display().to_string(),
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
            "verify": { "summary": verify_summary, "raw": if include_raw_verify { verify_raw_v } else { serde_json::Value::Null } },
        }))
    }
}

struct ProofpatchPatchNearestTool;

#[async_trait]
impl Tool for ProofpatchPatchNearestTool {
    fn description(&self) -> &str {
        "Patch the primary/nearest `sorry` in a file (no lemma/line args) and verify."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "repo_root": { "type": "string" },
                "file": { "type": "string" },
                "replacement": { "type": "string", "description": "Lean proof-term text to splice in (no markdown fences)" },
                "timeout_s": { "type": "integer", "default": 120 },
                "write": { "type": "boolean", "default": false, "description": "If true, write the patched text back to the file." },
                "max_sorries": { "type": "integer", "default": 50 },
                "context_lines": { "type": "integer", "default": 1 },
                "include_raw_verify": {
                    "type": "boolean",
                    "default": false,
                    "description": "If true, include full verify raw output (can be large)."
                }
            },
            "required": ["repo_root", "file", "replacement"]
        })
    }

    async fn call(&self, args: &Value) -> Result<Value, String> {
        let repo_root = repo_root_from_args(args)?;
        let file = extract_string(args, "file")?;
        let replacement = extract_string(args, "replacement")?;
        let timeout_s = extract_u64_opt(args, "timeout_s")?.unwrap_or(120);
        let write = args.get("write").and_then(|v| v.as_bool()).unwrap_or(false);
        let max_sorries = extract_u64_opt(args, "max_sorries")?.unwrap_or(50) as usize;
        let context_lines = extract_u64_opt(args, "context_lines")?.unwrap_or(1) as usize;
        let include_raw_verify = args
            .get("include_raw_verify")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let repo_root = resolve_lean_repo_root(repo_root, Some(&file))?;
        plc::load_dotenv_smart(&repo_root);

        let p = repo_root.join(&file);
        if !p.exists() {
            return Err(format!("File not found: {}", p.display()));
        }
        let original = std::fs::read_to_string(&p)
            .map_err(|e| format!("failed to read {}: {}", p.display(), e))?;

        let baseline =
            plc::verify_lean_file(&repo_root, &file, StdDuration::from_secs(timeout_s)).await?;
        let baseline_raw_v = serde_json::to_value(baseline)
            .map_err(|e| format!("failed to serialize verify result: {}", e))?;
        let baseline_summary = summarize_verify_like_output(&baseline_raw_v);
        let first_error_line_1 = baseline_summary
            .get("first_error_loc")
            .and_then(|v| v.get("line"))
            .and_then(|v| v.as_u64())
            .map(|x| x as usize);

        let locs = plc::locate_sorries_in_text(&original, max_sorries, context_lines)?;
        let selected = plc::select_primary_sorry(first_error_line_1, &locs).ok_or_else(|| {
            format!(
                "No `sorry`/`admit` tokens found (max_sorries={}, context_lines={}).",
                max_sorries, context_lines
            )
        })?;

        let patched = plc::patch_first_sorry_in_region(
            &original,
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
            std::fs::write(&p, patched.text.as_bytes())
                .map_err(|e| format!("failed to write {}: {}", p.display(), e))?;
            written_file = Some(p.display().to_string());
        }

        let verify = if write {
            plc::verify_lean_file(&repo_root, &file, StdDuration::from_secs(timeout_s)).await?
        } else {
            plc::verify_lean_text(&repo_root, &patched.text, StdDuration::from_secs(timeout_s))
                .await?
        };
        let verify_raw_v = serde_json::to_value(verify)
            .map_err(|e| format!("failed to serialize verify result: {}", e))?;
        let verify_summary = summarize_verify_like_output(&verify_raw_v);
        let selected_v =
            serde_json::to_value(&selected).map_err(|e| format!("failed to serialize sorry: {}", e))?;

        Ok(json!({
            "repo_root": repo_root.display().to_string(),
            "file": p.display().to_string(),
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
            "verify": { "summary": verify_summary, "raw": if include_raw_verify { verify_raw_v } else { serde_json::Value::Null } },
        }))
    }
}

struct ProofpatchTreeSearchNearestTool;

#[async_trait]
impl Tool for ProofpatchTreeSearchNearestTool {
    fn description(&self) -> &str {
        "Beam-search over candidate patches for successive `sorry`s (nearest-first), verifying each node."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "repo_root": { "type": "string" },
                "file": { "type": "string" },
                "timeout_s": { "type": "integer", "default": 120, "description": "Per-verify timeout (seconds)." },
                "beam": { "type": "integer", "default": 4, "description": "Beam width (kept small for boundedness)." },
                "max_nodes": { "type": "integer", "default": 20, "description": "Max nodes evaluated." },
                "depth": { "type": "integer", "default": 2, "description": "Max patch depth (number of sorries to try patching)." },
                "candidates_mode": {
                    "type": "string",
                    "default": "det",
                    "description": "Candidate selection mode: det (built-in), auto (derive from goal dump), lean (use simp?/exact?/apply? suggestions), or llm (JSON array from LLM)."
                },
                "include_goal_dump": {
                    "type": "boolean",
                    "default": false,
                    "description": "If true, run a goal-dump pass on the nearest sorry and include it in output."
                },
                "llm_timeout_s": {
                    "type": "integer",
                    "default": 60,
                    "description": "LLM timeout (seconds) when candidates_mode=llm."
                },
                "candidates": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Candidate replacements (proof terms). If omitted, uses a small built-in tactic list."
                },
                "allow_sorry_candidates": {
                    "type": "boolean",
                    "default": false,
                    "description": "If false (default), filter out candidates containing `sorry`/`admit`."
                },
                "include_trace": {
                    "type": "boolean",
                    "default": false,
                    "description": "If true, include the full search trace (can be large)."
                },
                "research_notes": {
                    "type": "string",
                    "description": "Optional external notes (e.g., ArXiv summary) to condition LLM candidate generation. Only used when candidates_mode=llm."
                },
                "write": { "type": "boolean", "default": false, "description": "If true, write best text back to the file." },
                "include_raw_verify": { "type": "boolean", "default": false, "description": "If true, include full verify raw output (can be large)." }
            },
            "required": ["repo_root", "file"]
        })
    }

    async fn call(&self, args: &Value) -> Result<Value, String> {
        #[derive(Clone)]
        struct Node {
            id: usize,
            parent: Option<usize>,
            depth: usize,
            text: String,
            last_region: Option<(usize, usize)>,
            last_replacement: Option<String>,
            verify_raw: serde_json::Value,
            verify_summary: serde_json::Value,
            sorries: usize,
            conservative_sorries: usize,
        }

        use plc::tree_search::{
            adapt_candidates_for_error,
            default_det_candidates as default_candidates,
            parse_json_string_array,
            progress_score_key,
            sanitize_candidates,
            verify_score_key as score_key,
        };

        let repo_root = repo_root_from_args(args)?;
        let file = extract_string(args, "file")?;
        let timeout_s = extract_u64_opt(args, "timeout_s")?.unwrap_or(120);
        let beam = extract_u64_opt(args, "beam")?.unwrap_or(4) as usize;
        let max_nodes = extract_u64_opt(args, "max_nodes")?.unwrap_or(20) as usize;
        let depth = extract_u64_opt(args, "depth")?.unwrap_or(2) as usize;
        let candidates_mode = args
            .get("candidates_mode")
            .and_then(|v| v.as_str())
            .unwrap_or("det")
            .trim()
            .to_lowercase();
        let include_goal_dump = args
            .get("include_goal_dump")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let llm_timeout_s = extract_u64_opt(args, "llm_timeout_s")?.unwrap_or(60);
        let allow_sorry_candidates = args
            .get("allow_sorry_candidates")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let include_trace = args
            .get("include_trace")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let research_notes = args
            .get("research_notes")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let write = args.get("write").and_then(|v| v.as_bool()).unwrap_or(false);
        let include_raw_verify = args
            .get("include_raw_verify")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let candidates_override = args
            .get("candidates")
            .and_then(|v| v.as_array())
            .map(|xs| {
                xs.iter()
                    .filter_map(|x| x.as_str().map(|s| s.trim().to_string()))
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
            })
            .filter(|xs: &Vec<String>| !xs.is_empty())
            ;

        if beam == 0 {
            return Err("beam must be >= 1".to_string());
        }
        if max_nodes == 0 {
            return Err("max_nodes must be >= 1".to_string());
        }
        if depth == 0 {
            return Err("depth must be >= 1".to_string());
        }

        let repo_root = resolve_lean_repo_root(repo_root, Some(&file))?;
        plc::load_dotenv_smart(&repo_root);

        let p = repo_root.join(&file);
        if !p.exists() {
            return Err(format!("File not found: {}", p.display()));
        }
        let original = std::fs::read_to_string(&p)
            .map_err(|e| format!("failed to read {}: {}", p.display(), e))?;

        let mut goal_dump_v: Option<serde_json::Value> = None;
        if include_goal_dump || candidates_mode == "auto" || candidates_mode == "llm" || candidates_mode == "lean" {
            if let Ok(gd) =
                plc::goal_dump_nearest(&repo_root, &file, StdDuration::from_secs(timeout_s)).await
            {
                goal_dump_v = Some(gd);
            }
        }

        let mut candidates = if let Some(ref xs) = candidates_override {
            xs.clone()
        } else if candidates_mode == "lean" {
            let ls = plc::lean_suggest_nearest(&repo_root, &file, StdDuration::from_secs(timeout_s)).await.ok();
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
                // fallback to goal-derived candidates when available
                let pretty = goal_dump_v
                    .as_ref()
                    .and_then(|gd| gd.get("pp_dump"))
                    .and_then(|v| v.get("goals"))
                    .and_then(|v| v.as_array())
                    .and_then(|xs| xs.first())
                    .and_then(|v| v.get("pretty"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let derived = plc::derive_candidates_from_goal_pretty(pretty);
                if derived.is_empty() { default_candidates() } else { derived }
            } else {
                xs
            }
        } else if candidates_mode == "auto" {
            let pretty = goal_dump_v
                .as_ref()
                .and_then(|gd| gd.get("pp_dump"))
                .and_then(|v| v.get("goals"))
                .and_then(|v| v.as_array())
                .and_then(|xs| xs.first())
                .and_then(|v| v.get("pretty"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let derived = plc::derive_candidates_from_goal_pretty(pretty);
            if derived.is_empty() { default_candidates() } else { derived }
        } else {
            default_candidates()
        };
        if candidates_mode == "llm" && candidates_override.is_none() {
            // Ask the LLM for a JSON array of candidates (bounded).
            let locs0 = plc::locate_sorries_in_text(&original, 50, 1).unwrap_or_default();
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
            system.push_str("\n\nReturn a JSON array of 6 distinct candidate Lean replacements (strings). Each element must be a proof term only (no markdown fences).");
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
            if let Some(notes) = research_notes.as_ref() {
                let max_chars = 12_000usize;
                let kept: String = notes.chars().take(max_chars).collect();
                system.push_str("\n\nResearch notes (external; may be incomplete):\n");
                system.push_str(&kept);
                if notes.chars().count() > max_chars {
                    system.push_str("\n\n[proofpatch: research_notes truncated]\n");
                }
            }
            let res = plc::llm::chat_completion(
                &system,
                &payload.user,
                StdDuration::from_secs(llm_timeout_s),
            ).await;
            if let Ok(done) = res {
                if let Some(xs) = parse_json_string_array(&done.content) {
                    candidates = xs;
                }
            }
        }
        candidates = sanitize_candidates(candidates);
        if !allow_sorry_candidates {
            candidates.retain(|c| {
                let lc = c.to_lowercase();
                !lc.contains("sorry") && !lc.contains("admit")
            });
            if candidates.is_empty() {
                candidates = sanitize_candidates(default_candidates());
            }
        }

        let baseline =
            plc::verify_lean_file(&repo_root, &file, StdDuration::from_secs(timeout_s)).await?;
        let baseline_raw_v = serde_json::to_value(baseline)
            .map_err(|e| format!("failed to serialize verify result: {}", e))?;
        let baseline_summary = summarize_verify_like_output(&baseline_raw_v);

        let mut next_id = 1usize;
        let mut trace: Vec<serde_json::Value> = Vec::new();
        let mut frontier: Vec<Node> = Vec::new();

        // Root node (unpatched).
        {
            let locs0 = plc::locate_sorries_in_text(&original, 500, 1).unwrap_or_default();
            let conservative0 = plc::count_sorry_tokens_conservative(&original).unwrap_or(0);
            frontier.push(Node {
                id: 0,
                parent: None,
                depth: 0,
                text: original.clone(),
                last_region: None,
                last_replacement: None,
                verify_raw: baseline_raw_v.clone(),
                verify_summary: baseline_summary.clone(),
                sorries: locs0.len(),
                conservative_sorries: conservative0,
            });
        }

        let mut best_done: Option<Node> = None;

        while !frontier.is_empty() && trace.len() < max_nodes {
            for n in frontier.iter() {
                let ok = n.verify_summary.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
                if ok && n.sorries == 0 {
                    best_done = Some(n.clone());
                    break;
                }
            }
            if best_done.is_some() {
                break;
            }

            frontier.sort_by(|a, b| {
                score_key(&a.verify_summary, a.sorries, a.conservative_sorries)
                    .cmp(&score_key(&b.verify_summary, b.sorries, b.conservative_sorries))
                    .then_with(|| a.id.cmp(&b.id))
            });
            if frontier.len() > beam {
                frontier.truncate(beam);
            }

            for n in frontier.iter() {
                if trace.len() >= max_nodes {
                    break;
                }
                trace.push(json!({
                    "id": n.id,
                    "parent": n.parent,
                    "depth": n.depth,
                    "last_region": n.last_region.map(|(a,b)| json!({"start_line": a, "end_line": b})),
                    "last_replacement": n.last_replacement,
                    "sorries": n.sorries,
                    "conservative_sorries": n.conservative_sorries,
                    "verify": { "summary": n.verify_summary, "raw": if include_raw_verify { n.verify_raw.clone() } else { serde_json::Value::Null } },
                }));
            }
            if trace.len() >= max_nodes {
                break;
            }

            let mut new_frontier: Vec<Node> = Vec::new();
            for parent in frontier.iter() {
                if parent.depth >= depth {
                    continue;
                }
                let first_error_line_1 = parent
                    .verify_summary
                    .get("first_error_loc")
                    .and_then(|v| v.get("line"))
                    .and_then(|v| v.as_u64())
                    .map(|x| x as usize);
                let locs = plc::locate_sorries_in_text(&parent.text, 50, 1).unwrap_or_default();
                let Some(sel) = plc::select_primary_sorry(first_error_line_1, &locs) else {
                    continue;
                };
                let region = (sel.region_start, sel.region_end);
                let parent_first_error = parent
                    .verify_summary
                    .get("first_error")
                    .and_then(|v| v.as_str());
                let candidates_here = adapt_candidates_for_error(&candidates, parent_first_error);

                for cand in candidates_here.iter() {
                    if trace.len() + new_frontier.len() >= max_nodes {
                        break;
                    }
                    let Ok(patched) = plc::patch_first_sorry_in_region(&parent.text, region.0, region.1, cand) else {
                        continue;
                    };
                    let raw = plc::verify_lean_text(&repo_root, &patched.text, StdDuration::from_secs(timeout_s)).await?;
                    let raw_v = serde_json::to_value(raw)
                        .map_err(|e| format!("failed to serialize verify result: {}", e))?;
                    let summary = summarize_verify_like_output(&raw_v);
                    let locs2 = plc::locate_sorries_in_text(&patched.text, 500, 1).unwrap_or_default();
                    let conservative2 = plc::count_sorry_tokens_conservative(&patched.text).unwrap_or(0);
                    new_frontier.push(Node {
                        id: next_id,
                        parent: Some(parent.id),
                        depth: parent.depth + 1,
                        text: patched.text.clone(),
                        last_region: Some(region),
                        last_replacement: Some(cand.clone()),
                        verify_raw: raw_v,
                        verify_summary: summary,
                        sorries: locs2.len(),
                        conservative_sorries: conservative2,
                    });
                    next_id += 1;
                }
            }

            frontier = new_frontier;
        }

        // Pick best (prefer completed; else best remaining).
        let mut best = best_done;
        if best.is_none() && !frontier.is_empty() {
            frontier.sort_by(|a, b| {
                score_key(&a.verify_summary, a.sorries, a.conservative_sorries)
                    .cmp(&score_key(&b.verify_summary, b.sorries, b.conservative_sorries))
                    .then_with(|| a.id.cmp(&b.id))
            });
            best = Some(frontier[0].clone());
        }
        let Some(best) = best else {
            return Err("tree_search_nearest: no nodes evaluated".to_string());
        };

        // Compute "best progress" from the recorded trace (best = score_key; progress = fewer sorries).
        let best_progress = if trace.is_empty() {
            best.clone()
        } else {
            let mut best_idx = 0usize;
            let mut best_key = (i64::MAX, 9, i64::MAX, i64::MAX);
            for (i, v) in trace.iter().enumerate() {
                let summary = v.get("verify").and_then(|x| x.get("summary")).unwrap_or(&serde_json::Value::Null);
                let sorries = v.get("sorries").and_then(|x| x.as_u64()).unwrap_or(999) as usize;
                let conservative = v.get("conservative_sorries").and_then(|x| x.as_u64()).unwrap_or(999) as usize;
                let k = progress_score_key(summary, sorries, conservative);
                if k < best_key {
                    best_key = k;
                    best_idx = i;
                }
            }
            // Reconstruct a minimal "node-like" view from trace for output.
            let chosen = &trace[best_idx];
            Node {
                id: chosen.get("id").and_then(|x| x.as_u64()).unwrap_or(0) as usize,
                parent: chosen.get("parent").and_then(|x| x.as_u64()).map(|x| x as usize),
                depth: chosen.get("depth").and_then(|x| x.as_u64()).unwrap_or(0) as usize,
                text: best.text.clone(),
                last_region: None,
                last_replacement: None,
                verify_raw: serde_json::Value::Null,
                verify_summary: chosen.get("verify").and_then(|x| x.get("summary")).cloned().unwrap_or(serde_json::Value::Null),
                sorries: chosen.get("sorries").and_then(|x| x.as_u64()).unwrap_or(999) as usize,
                conservative_sorries: chosen.get("conservative_sorries").and_then(|x| x.as_u64()).unwrap_or(999) as usize,
            }
        };

        let mut written_file: Option<String> = None;
        if write {
            std::fs::write(&p, best.text.as_bytes())
                .map_err(|e| format!("failed to write {}: {}", p.display(), e))?;
            written_file = Some(p.display().to_string());
        }

        Ok(json!({
            "repo_root": repo_root.display().to_string(),
            "file": p.display().to_string(),
            "written_file": written_file,
            "config": {
                "timeout_s": timeout_s,
                "beam": beam,
                "max_nodes": max_nodes,
                "depth": depth,
                "candidates_mode": candidates_mode,
                "candidates_count": candidates.len(),
                "allow_sorry_candidates": allow_sorry_candidates,
                "include_trace": include_trace
            },
            "baseline_verify": { "summary": baseline_summary },
            "goal_dump": goal_dump_v,
            "best": {
                "id": best.id,
                "parent": best.parent,
                "depth": best.depth,
                "last_region": best.last_region.map(|(a,b)| json!({"start_line": a, "end_line": b})),
                "last_replacement": best.last_replacement,
                "sorries": best.sorries,
                "conservative_sorries": best.conservative_sorries,
                "verify": { "summary": best.verify_summary, "raw": if include_raw_verify { best.verify_raw } else { serde_json::Value::Null } },
            },
            "best_progress": {
                "id": best_progress.id,
                "parent": best_progress.parent,
                "depth": best_progress.depth,
                "sorries": best_progress.sorries,
                "conservative_sorries": best_progress.conservative_sorries,
                "verify": { "summary": best_progress.verify_summary }
            },
            "trace": if include_trace { serde_json::Value::Array(trace) } else { serde_json::Value::Null }
        }))
    }
}

struct ProofpatchLocateSorriesTool;

#[async_trait]
impl Tool for ProofpatchLocateSorriesTool {
    fn description(&self) -> &str {
        "Locate `sorry` tokens in a file with line/col and suggested patch regions."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "repo_root": { "type": "string" },
                "file": { "type": "string" },
                "max_results": { "type": "integer", "default": 50 },
                "context_lines": { "type": "integer", "default": 2 }
            },
            "required": ["repo_root", "file"]
        })
    }

    async fn call(&self, args: &Value) -> Result<Value, String> {
        let file = extract_string(args, "file")?;
        let max_results = extract_u64_opt(args, "max_results")?.unwrap_or(50) as usize;
        let context_lines = extract_u64_opt(args, "context_lines")?.unwrap_or(2) as usize;

        let repo_root = repo_root_from_args(args)?;
        let repo_root = resolve_lean_repo_root(repo_root, Some(&file))?;
        let locs = plc::locate_sorries_in_file(&repo_root, &file, max_results, context_lines)?;

        Ok(json!({
            "repo_root": repo_root.display().to_string(),
            "file": file,
            "count": locs.len(),
            "locations": locs,
        }))
    }
}

struct ProofpatchContextPackTool;

#[async_trait]
impl Tool for ProofpatchContextPackTool {
    fn description(&self) -> &str {
        "Build a JSON-first context pack for a file + decl/line (imports + excerpt + nearby decls)."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "repo_root": { "type": "string" },
                "file": { "type": "string" },
                "decl": { "type": "string", "description": "Optional decl name to focus (theorem/lemma/def/etc)." },
                "line": { "type": "integer", "description": "Optional 1-based line number to focus." },
                "context_lines": { "type": "integer", "default": 25, "description": "Line-window radius when focusing by line." },
                "nearby_lines": { "type": "integer", "default": 80, "description": "How far around the focus to scan for nearby decl headers." },
                "max_nearby_decls": { "type": "integer", "default": 30 },
                "max_imports": { "type": "integer", "default": 50 }
            },
            "required": ["repo_root", "file"]
        })
    }

    async fn call(&self, args: &Value) -> Result<Value, String> {
        let file = extract_string(args, "file")?;
        let decl = extract_string_opt(args, "decl");
        let line = extract_u64_opt(args, "line")?.map(|x| x as usize);
        let context_lines = extract_u64_opt(args, "context_lines")?.unwrap_or(25) as usize;
        let nearby_lines = extract_u64_opt(args, "nearby_lines")?.unwrap_or(80) as usize;
        let max_nearby_decls = extract_u64_opt(args, "max_nearby_decls")?.unwrap_or(30) as usize;
        let max_imports = extract_u64_opt(args, "max_imports")?.unwrap_or(50) as usize;

        let repo_root = repo_root_from_args(args)?;
        let repo_root = resolve_lean_repo_root(repo_root, Some(&file))?;
        let pack = plc::build_context_pack(
            &repo_root,
            &file,
            decl.as_deref(),
            line,
            context_lines,
            nearby_lines,
            max_nearby_decls,
            max_imports,
        )?;
        serde_json::to_value(pack).map_err(|e| format!("failed to serialize context pack: {}", e))
    }
}

struct ProofpatchTriageFileTool;

#[async_trait]
impl Tool for ProofpatchTriageFileTool {
    fn description(&self) -> &str {
        "Triage a file: verify_summary + locate_sorries, plus nearest sorry to first error (if any)."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "repo_root": { "type": "string" },
                "file": { "type": "string" },
                "timeout_s": { "type": "integer", "default": 180 },
                "max_sorries": { "type": "integer", "default": 50 },
                "context_lines": { "type": "integer", "default": 1 },
                "include_raw_verify": {
                    "type": "boolean",
                    "default": false,
                    "description": "If true, include full verify raw output (can be large)."
                },
                "include_context_pack": {
                    "type": "boolean",
                    "default": true,
                    "description": "If true, include context packs around first error + nearest sorry."
                },
                "pack_context_lines": { "type": "integer", "default": 6 },
                "pack_nearby_lines": { "type": "integer", "default": 60 },
                "pack_max_nearby_decls": { "type": "integer", "default": 20 },
                "pack_max_imports": { "type": "integer", "default": 20 },
                "include_prompts": {
                    "type": "boolean",
                    "default": true,
                    "description": "If true, include a rubberduck prompt for first error and a region-patch prompt for nearest sorry."
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional path to write the full JSON output. If set, response includes a small summary + `written_path`."
                }
            },
            "required": ["repo_root", "file"]
        })
    }

    async fn call(&self, args: &Value) -> Result<Value, String> {
        let file = extract_string(args, "file")?;
        let repo_root = repo_root_from_args(args)?;
        let timeout_s = extract_u64_opt(args, "timeout_s")?.unwrap_or(180);
        let max_sorries = extract_u64_opt(args, "max_sorries")?.unwrap_or(50) as usize;
        let context_lines = extract_u64_opt(args, "context_lines")?.unwrap_or(1) as usize;
        let include_raw_verify = args
            .get("include_raw_verify")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let include_context_pack = args
            .get("include_context_pack")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let pack_context_lines = extract_u64_opt(args, "pack_context_lines")?.unwrap_or(6) as usize;
        let pack_nearby_lines = extract_u64_opt(args, "pack_nearby_lines")?.unwrap_or(60) as usize;
        let pack_max_nearby_decls =
            extract_u64_opt(args, "pack_max_nearby_decls")?.unwrap_or(20) as usize;
        let pack_max_imports = extract_u64_opt(args, "pack_max_imports")?.unwrap_or(20) as usize;
        let include_prompts = args
            .get("include_prompts")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let output_path = extract_string_opt(args, "output_path");

        let repo_root = resolve_lean_repo_root(repo_root, Some(&file))?;
        plc::load_dotenv_smart(&repo_root);

        let raw =
            plc::verify_lean_file(&repo_root, &file, StdDuration::from_secs(timeout_s)).await?;
        let raw_v = serde_json::to_value(raw)
            .map_err(|e| format!("failed to serialize verify result: {}", e))?;
        let summary = summarize_verify_like_output(&raw_v);

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
                        pack_max_nearby_decls,
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
                        pack_max_nearby_decls,
                        pack_max_imports,
                    )
                    .ok()
                })
                .and_then(|p| serde_json::to_value(p).ok());

            (pack_first, pack_nearest)
        } else {
            (None, None)
        };

        let first_error_text = summary
            .get("first_error")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let rubberduck_prompt_first_error = if include_prompts {
            context_pack_first_error
                .as_ref()
                .and_then(|v| v.get("focus"))
                .and_then(|f| f.get("excerpt"))
                .and_then(|e| e.as_str())
                .and_then(|excerpt| {
                    let label = summary
                        .get("first_error_loc")
                        .and_then(|v| v.get("line"))
                        .and_then(|v| v.as_u64())
                        .map(|l| format!("line:{l}"))
                        .unwrap_or_else(|| "line:?".to_string());
                    plc::build_rubberduck_prompt_from_excerpt(
                        &repo_root,
                        &file,
                        &label,
                        excerpt,
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
            let first_error_line = summary
                .get("first_error_loc")
                .and_then(|v| v.get("line"))
                .and_then(|v| v.as_u64());

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

        let full = json!({
            "repo_root": repo_root.display().to_string(),
            "file": file,
            "verify": { "summary": summary, "raw": if include_raw_verify { raw_v } else { serde_json::Value::Null } },
            "sorries": { "count": locs.len(), "conservative_count": conservative_sorries, "locations": locs },
            "nearest_sorry_to_first_error": selected_sorry,
            "context_pack_first_error": context_pack_first_error,
            "context_pack_nearest_sorry": context_pack_nearest_sorry,
            "rubberduck_prompt_first_error": rubberduck_prompt_first_error,
            "patch_prompt_nearest_sorry": patch_prompt_nearest_sorry,
            "next_action": next_action,
        });

        if let Some(p) = output_path {
            let path = std::path::PathBuf::from(&p);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    format!("failed to create output dir {}: {}", parent.display(), e)
                })?;
            }
            let s = serde_json::to_string_pretty(&full)
                .map_err(|e| format!("failed to encode json: {}", e))?;
            std::fs::write(&path, s.as_bytes())
                .map_err(|e| format!("failed to write {}: {}", path.display(), e))?;

            let small = json!({
                "ok": true,
                "written_path": path.display().to_string(),
                "repo_root": repo_root.display().to_string(),
                "file": full.get("file").cloned().unwrap_or(serde_json::Value::Null),
                "verify_ok": summary.get("ok").cloned().unwrap_or(serde_json::Value::Null),
                "errors": summary.get("counts").and_then(|c| c.get("errors")).cloned().unwrap_or(serde_json::Value::Null),
                "sorries": locs.len(),
                "next_action": full.get("next_action").cloned().unwrap_or(serde_json::Value::Null),
            });
            return Ok(small);
        }

        Ok(full)
    }
}

fn apply_mechanical_fixes_for_first_error(
    text: &str,
    first_error_line_1: Option<usize>,
    first_error_text: Option<&str>,
) -> (String, Vec<Value>) {
    let Some(line1) = first_error_line_1 else {
        return (text.to_string(), vec![]);
    };
    let Some(msg) = first_error_text else {
        return (text.to_string(), vec![]);
    };

    let mut lines: Vec<String> = text.lines().map(|s| s.to_string()).collect();
    if lines.is_empty() {
        return (text.to_string(), vec![]);
    }

    let mut edits = Vec::new();
    let start0 = line1.saturating_sub(1).saturating_sub(40);
    let end0 = usize::min(lines.len().saturating_sub(1), (line1 - 1) + 60);

    // Heuristic 1: if Lean suggests `ring_nf`, replace a nearby `ring` with `ring_nf`.
    if msg.contains("ring_nf") {
        for i0 in start0..=end0 {
            let ln = &lines[i0];
            if ln.trim() == "ring" {
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
    }

    // Heuristic 2: if the first error is a missing `Decidable` and we have a nearby `decide`,
    // insert `classical` before `decide`.
    //
    // This is common in mathlib code where a `decide` proof works under classical but Lean
    // cannot synthesize an instance automatically.
    if msg.contains("failed to synthesize") && msg.contains("Decidable") {
        for i0 in start0..=end0 {
            let ln = &lines[i0];
            if ln.trim() != "decide" {
                continue;
            }

            // Avoid duplicating `classical` if it's already in the local tactic block.
            let scan0 = i0.saturating_sub(4);
            let already_classical = (scan0..=i0)
                .any(|j0| lines.get(j0).map(|s| s.trim() == "classical").unwrap_or(false));
            if already_classical {
                continue;
            }

            let before = ln.clone();
            let indent: String = ln.chars().take_while(|c| c.is_whitespace()).collect();
            let replacement_lines = vec![format!("{indent}classical"), format!("{indent}decide")];
            lines.splice(i0..=i0, replacement_lines.clone());
            edits.push(json!({
                "kind": "insert_classical_before_decide",
                "line": i0 + 1,
                "before": before,
                "after": replacement_lines.join("\n"),
                "note": "First error looks like missing Decidable; inserted `classical` before a nearby `decide`.",
            }));
            break;
        }
    }

    let mut out = lines.join("\n");
    if text.ends_with('\n') {
        out.push('\n');
    }
    (out, edits)
}

fn first_error_snippet(stdout: &str, stderr: &str, max_lines: usize) -> Option<String> {
    fn from_text(txt: &str, max_lines: usize) -> Option<String> {
        let lines: Vec<&str> = txt.lines().collect();
        let i0 = lines.iter().position(|l| l.contains(": error:"))?;
        let end = usize::min(lines.len(), i0 + max_lines.max(1));
        Some(lines[i0..end].join("\n"))
    }

    from_text(stdout, max_lines).or_else(|| from_text(stderr, max_lines))
}

struct ProofpatchAgentStepTool;

#[async_trait]
impl Tool for ProofpatchAgentStepTool {
    fn description(&self) -> &str {
        "Execute one safe agent step (no LLM): verify → apply mechanical fix → verify."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "repo_root": { "type": "string" },
                "file": { "type": "string" },
                "timeout_s": { "type": "integer", "default": 180 },
                "write": { "type": "boolean", "default": false, "description": "If true, write edits back to the file. Otherwise verifies against a temp file." },
                "output_path": { "type": "string", "description": "Optional path to write full JSON output; response becomes a small summary." }
            },
            "required": ["repo_root", "file"]
        })
    }

    async fn call(&self, args: &Value) -> Result<Value, String> {
        let file = extract_string(args, "file")?;
        let repo_root = repo_root_from_args(args)?;
        let timeout_s = extract_u64_opt(args, "timeout_s")?.unwrap_or(180);
        let write = args.get("write").and_then(|v| v.as_bool()).unwrap_or(false);
        let output_path = extract_string_opt(args, "output_path");

        let repo_root = resolve_lean_repo_root(repo_root, Some(&file))?;
        plc::load_dotenv_smart(&repo_root);

        let abs = repo_root.join(&file);
        if !abs.exists() {
            return Err(format!("File not found: {}", abs.display()));
        }
        let original_text =
            std::fs::read_to_string(&abs).map_err(|e| format!("read {}: {e}", abs.display()))?;

        let verify0 =
            plc::verify_lean_file(&repo_root, &file, StdDuration::from_secs(timeout_s)).await?;
        let first_error_loc = plc::parse_first_error_loc(&verify0.stdout, &verify0.stderr);
        let first_error_text =
            first_error_snippet(&verify0.stdout, &verify0.stderr, 12);

        let (patched_text, edits) = apply_mechanical_fixes_for_first_error(
            &original_text,
            first_error_loc.as_ref().map(|l| l.line),
            first_error_text.as_deref(),
        );

        let wrote_file = if write && !edits.is_empty() {
            std::fs::write(&abs, patched_text.as_bytes())
                .map_err(|e| format!("write {}: {e}", abs.display()))?;
            Some(abs.display().to_string())
        } else {
            None
        };

        let verify1 = if write && !edits.is_empty() {
            plc::verify_lean_file(&repo_root, &file, StdDuration::from_secs(timeout_s)).await?
        } else {
            plc::verify_lean_text(&repo_root, &patched_text, StdDuration::from_secs(timeout_s))
                .await?
        };

        let dag = json!({
            "nodes": [
                { "id": "verify0", "kind": "verify", "ok": verify0.ok, "returncode": verify0.returncode, "first_error_loc": first_error_loc, "first_error": first_error_text },
                { "id": "mech_fix1", "kind": "mechanical_fix", "applied": !edits.is_empty(), "edits": edits, "write": write },
                { "id": "verify1", "kind": "verify", "ok": verify1.ok, "returncode": verify1.returncode, "first_error_loc": plc::parse_first_error_loc(&verify1.stdout, &verify1.stderr), "first_error": verify1.stdout.lines().find(|l| l.contains(": error:")) }
            ],
            "edges": [
                { "from": "verify0", "to": "mech_fix1" },
                { "from": "mech_fix1", "to": "verify1" }
            ]
        });

        let full = json!({
            "ok": true,
            "repo_root": repo_root.display().to_string(),
            "file": file,
            "written_file": wrote_file,
            "dag": dag
        });

        if let Some(p) = output_path {
            let path = std::path::PathBuf::from(&p);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    format!("failed to create output dir {}: {}", parent.display(), e)
                })?;
            }
            let s = serde_json::to_string_pretty(&full)
                .map_err(|e| format!("failed to encode json: {}", e))?;
            std::fs::write(&path, s.as_bytes())
                .map_err(|e| format!("failed to write {}: {}", path.display(), e))?;

            return Ok(json!({
                "ok": true,
                "written_path": path.display().to_string(),
                "file": file,
                "written_file": wrote_file,
                "verify1_ok": verify1.ok
            }));
        }

        Ok(full)
    }
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

struct ProofpatchReportHtmlTool;

#[async_trait]
impl Tool for ProofpatchReportHtmlTool {
    fn description(&self) -> &str {
        "Triage many files and write a small HTML report."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "repo_root": { "type": "string" },
                "files": { "type": "array", "items": { "type": "string" } },
                "timeout_s": { "type": "integer", "default": 180 },
                "max_sorries": { "type": "integer", "default": 5 },
                "context_lines": { "type": "integer", "default": 1 },
                "include_raw_verify": {
                    "type": "boolean",
                    "default": false,
                    "description": "If true, include full verify raw output per file (can be large)."
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional path to write the HTML report. If omitted, writes to a temp file."
                }
            },
            "required": ["repo_root", "files"]
        })
    }

    async fn call(&self, args: &Value) -> Result<Value, String> {
        let repo_root = repo_root_from_args(args)?;
        let timeout_s = extract_u64_opt(args, "timeout_s")?.unwrap_or(180);
        let max_sorries = extract_u64_opt(args, "max_sorries")?.unwrap_or(5) as usize;
        let context_lines = extract_u64_opt(args, "context_lines")?.unwrap_or(1) as usize;
        let include_raw_verify = args
            .get("include_raw_verify")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let output_path = extract_string_opt(args, "output_path");

        let files_v = args
            .get("files")
            .and_then(|v| v.as_array())
            .ok_or_else(|| "missing/invalid `files` (expected array of strings)".to_string())?;
        let mut files: Vec<String> = Vec::with_capacity(files_v.len());
        for v in files_v {
            let s = v
                .as_str()
                .ok_or_else(|| "invalid `files` entry (expected string)".to_string())?;
            files.push(s.to_string());
        }

        let repo_root = resolve_lean_repo_root(repo_root, files.get(0).map(|s| s.as_str()))?;
        plc::load_dotenv_smart(&repo_root);

        let mut items: Vec<Value> = Vec::with_capacity(files.len());
        let mut table: Vec<Value> = Vec::with_capacity(files.len());
        for file in &files {
            let raw =
                plc::verify_lean_file(&repo_root, file, StdDuration::from_secs(timeout_s)).await?;
            let raw_v = serde_json::to_value(raw)
                .map_err(|e| format!("failed to serialize verify result: {}", e))?;
            let summary = summarize_verify_like_output(&raw_v);

            let locs = plc::locate_sorries_in_file(&repo_root, file, max_sorries, context_lines)?;
            let conservative_sorries =
                plc::count_sorry_tokens_conservative_in_file(&repo_root, file).unwrap_or(0);

            let ok = summary.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
            let counts = summary.get("counts").cloned().unwrap_or_else(|| json!({}));
            let errors = counts.get("errors").and_then(|v| v.as_u64()).unwrap_or(0);
            let warnings = counts.get("warnings").and_then(|v| v.as_u64()).unwrap_or(0);

            table.push(json!({
                "file": file,
                "verify": { "ok": ok, "errors": errors, "warnings": warnings },
                "sorries": { "count": locs.len(), "conservative_count": conservative_sorries, "locations": locs },
            }));

            items.push(json!({
                "file": file,
                "verify": if include_raw_verify {
                    json!({ "summary": summary, "raw": raw_v })
                } else {
                    json!({ "summary": summary })
                },
                "sorries": { "count": locs.len(), "conservative_count": conservative_sorries, "locations": locs },
            }));
        }

        let report_path = match output_path {
            Some(p) => std::path::PathBuf::from(p),
            None => std::env::temp_dir()
                .join(format!("proofpatch-report-{}.html", uuid::Uuid::new_v4())),
        };
        if let Some(parent) = report_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create report dir {}: {}", parent.display(), e))?;
        }

        let mut html = String::new();
        html.push_str("<!doctype html>\n<html><head><meta charset=\"utf-8\"/>\n");
        html.push_str("<title>proofpatch report</title>\n");
        html.push_str(
            "<style>body{font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial;max-width:1200px;margin:24px auto;padding:0 16px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px;vertical-align:top}th{background:#f6f6f6;text-align:left}code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,\"Liberation Mono\",monospace}pre{white-space:pre-wrap}</style>\n",
        );
        html.push_str("</head><body>\n");
        html.push_str("<h2>proofpatch report</h2>\n");
        html.push_str(&format!(
            "<p><b>repo_root</b>: <code>{}</code></p>\n",
            escape_html(&repo_root.display().to_string())
        ));
        html.push_str(&format!("<p><b>files</b>: {}</p>\n", files.len()));

        html.push_str("<table>\n<thead><tr><th>file</th><th>verify</th><th>sorries</th></tr></thead>\n<tbody>\n");

        for item in &items {
            let file = item.get("file").and_then(|v| v.as_str()).unwrap_or("");
            let verify_summary = item
                .get("verify")
                .and_then(|v| v.get("summary"))
                .cloned()
                .unwrap_or_else(|| json!({}));
            let sorries = item
                .get("sorries")
                .and_then(|v| v.get("locations"))
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();

            let ok = verify_summary
                .get("ok")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let counts = verify_summary
                .get("counts")
                .cloned()
                .unwrap_or_else(|| json!({}));
            let errors = counts.get("errors").and_then(|v| v.as_u64()).unwrap_or(0);
            let warnings = counts.get("warnings").and_then(|v| v.as_u64()).unwrap_or(0);

            html.push_str("<tr>");
            html.push_str(&format!("<td><code>{}</code></td>", escape_html(file)));
            html.push_str(&format!(
                "<td><b>ok</b>: {}<br/><b>errors</b>: {}<br/><b>warnings</b>: {}</td>",
                ok, errors, warnings
            ));

            html.push_str("<td>");
            html.push_str(&format!("<b>count</b>: {}<br/>", sorries.len()));
            for loc in sorries.iter().take(5) {
                let line = loc.get("line").and_then(|v| v.as_u64()).unwrap_or(0);
                let col = loc.get("col").and_then(|v| v.as_u64()).unwrap_or(0);
                let excerpt = loc.get("excerpt").and_then(|v| v.as_str()).unwrap_or("");
                html.push_str(&format!(
                    "<div><b>@</b> {}:{}<pre>{}</pre></div>",
                    line,
                    col,
                    escape_html(excerpt)
                ));
            }
            html.push_str("</td>");
            html.push_str("</tr>\n");
        }

        html.push_str("</tbody>\n</table>\n");
        html.push_str("</body></html>\n");

        std::fs::write(&report_path, html.as_bytes())
            .map_err(|e| format!("failed to write report {}: {}", report_path.display(), e))?;

        Ok(json!({
            "repo_root": repo_root.display().to_string(),
            "report_path": report_path.display().to_string(),
            "table": table,
            "items": items,
        }))
    }
}

struct ProofpatchRubberduckPromptTool;

#[async_trait]
impl Tool for ProofpatchRubberduckPromptTool {
    fn description(&self) -> &str {
        "Build a rubberduck/ideation prompt for a lemma (no proof code; plan + next moves)."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "repo_root": { "type": "string" },
                "file": { "type": "string" },
                "lemma": { "type": "string" },
                "diagnostics": { "type": "string", "description": "Optional Lean output/error context to include (raw stdout/stderr excerpt or JSON)" },
                "proofpatch_root": { "type": "string" }
            },
            "required": ["repo_root", "file", "lemma"]
        })
    }

    async fn call(&self, args: &Value) -> Result<Value, String> {
        let repo_root = repo_root_from_args(args)?;
        let file = extract_string(args, "file")?;
        let lemma = extract_string(args, "lemma")?;
        let diagnostics = extract_string_opt(args, "diagnostics");
        let payload =
            plc::build_rubberduck_prompt(&repo_root, &file, &lemma, diagnostics.as_deref())?;
        serde_json::to_value(payload).map_err(|e| format!("failed to serialize payload: {}", e))
    }
}

struct ProofpatchLoopTool;

#[async_trait]
impl Tool for ProofpatchLoopTool {
    fn description(&self) -> &str {
        "Bounded loop: suggest → patch first `sorry` in lemma → verify (`proofpatch loop`)."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "repo_root": { "type": "string" },
                "file": { "type": "string" },
                "lemma": { "type": "string" },
                "max_iters": { "type": "integer", "default": 3 },
                "timeout_s": { "type": "integer", "default": 120 },
                "proofpatch_root": { "type": "string" }
            },
            "required": ["repo_root", "file", "lemma"]
        })
    }

    async fn call(&self, args: &Value) -> Result<Value, String> {
        let file = extract_string(args, "file")?;
        let lemma = extract_string(args, "lemma")?;
        let repo_root = repo_root_from_args(args)?;
        let max_iters = extract_u64_opt(args, "max_iters")?.unwrap_or(3);
        let timeout_s = extract_u64_opt(args, "timeout_s")?.unwrap_or(120);

        // Rust-native loop for suggest + patch + verify.
        let _ = proofpatch_root_from_args(args)?;
        let repo_root = resolve_lean_repo_root(repo_root, Some(&file))?;
        plc::load_dotenv_smart(&repo_root);

        let p = repo_root.join(&file);
        if !p.exists() {
            return Err(format!("File not found: {}", p.display()));
        }
        let mut cur_text = std::fs::read_to_string(&p)
            .map_err(|e| format!("failed to read {}: {}", p.display(), e))?;

        let mut attempts: Vec<Value> = Vec::new();

        for iter_idx in 0..max_iters {
            // Build prompt from the *current* text (we don't write to disk during the loop).
            let excerpt = plc::extract_decl_block(&cur_text, &lemma)?;
            let system = plc::proof_system_prompt();
            let user = plc::proof_user_prompt(&excerpt);

            let res = plc::llm::chat_completion(&system, &user, StdDuration::from_secs(timeout_s))
                .await?;

            let suggestion = json!({
                "provider": res.provider,
                "model": res.model,
                "lemma": lemma,
                "file": p.display().to_string(),
                "suggestion": res.content,
                "raw": res.raw,
            });

            let replacement = suggestion
                .get("suggestion")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "LLM suggestion did not contain `suggestion` field".to_string())?;

            let patched = plc::patch_first_sorry_in_decl(&cur_text, &lemma, replacement)?;
            cur_text = patched.text.clone();

            let still_has_sorry = plc::decl_block_contains_sorry(&cur_text, &lemma)?;
            let verify =
                plc::verify_lean_text(&repo_root, &cur_text, StdDuration::from_secs(timeout_s))
                    .await?;

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
        Ok(json!({
            "file": p.display().to_string(),
            "lemma": lemma,
            "max_iters": max_iters,
            "attempts": attempts,
            "final_lemma_contains_sorry": final_still_has_sorry,
        }))
    }
}

// =========================
// stdio MCP (rmcp) service
// =========================

#[cfg(feature = "stdio")]
#[derive(Debug, Deserialize, JsonSchema)]
struct RepoFileSorriesArgs {
    repo_root: String,
    file: String,
    #[serde(default)]
    timeout_s: Option<u64>,
    #[serde(default)]
    max_sorries: Option<u64>,
    #[serde(default)]
    context_lines: Option<u64>,
    #[serde(default)]
    include_raw_verify: Option<bool>,
    #[serde(default)]
    include_context_pack: Option<bool>,
    #[serde(default)]
    pack_context_lines: Option<u64>,
    #[serde(default)]
    pack_nearby_lines: Option<u64>,
    #[serde(default)]
    pack_max_nearby_decls: Option<u64>,
    #[serde(default)]
    pack_max_imports: Option<u64>,
    #[serde(default)]
    include_prompts: Option<bool>,
    #[serde(default)]
    output_path: Option<String>,
}

#[cfg(feature = "stdio")]
#[derive(Debug, Deserialize, JsonSchema)]
struct ContextPackArgs {
    repo_root: String,
    file: String,
    #[serde(default)]
    decl: Option<String>,
    #[serde(default)]
    line: Option<u64>,
    #[serde(default)]
    context_lines: Option<u64>,
    #[serde(default)]
    nearby_lines: Option<u64>,
    #[serde(default)]
    max_nearby_decls: Option<u64>,
    #[serde(default)]
    max_imports: Option<u64>,
}

#[cfg(feature = "stdio")]
#[derive(Debug, Deserialize, JsonSchema)]
struct AgentStepArgs {
    repo_root: String,
    file: String,
    #[serde(default)]
    timeout_s: Option<u64>,
    #[serde(default)]
    write: Option<bool>,
    #[serde(default)]
    output_path: Option<String>,
}

#[cfg(feature = "stdio")]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct PromptArgs {
    repo_root: String,
    file: String,
    lemma: String,
    #[serde(default)]
    timeout_s: Option<u64>,
    // Schema compatibility only (unused).
    #[serde(default)]
    proofpatch_root: Option<String>,
}

#[cfg(feature = "stdio")]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct VerifyArgs {
    repo_root: String,
    file: String,
    #[serde(default)]
    timeout_s: Option<u64>,
    // Schema compatibility only (unused).
    #[serde(default)]
    proofpatch_root: Option<String>,
}

#[cfg(feature = "stdio")]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct LocateSorriesArgs {
    repo_root: String,
    file: String,
    #[serde(default)]
    max_results: Option<u64>,
    #[serde(default)]
    context_lines: Option<u64>,
}

#[cfg(feature = "stdio")]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct PatchRegionArgs {
    repo_root: String,
    file: String,
    start_line: u64,
    end_line: u64,
    replacement: String,
    #[serde(default)]
    timeout_s: Option<u64>,
}

#[cfg(feature = "stdio")]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct PatchArgs {
    repo_root: String,
    file: String,
    lemma: String,
    replacement: String,
    #[serde(default)]
    timeout_s: Option<u64>,
    // Schema compatibility only (unused).
    #[serde(default)]
    proofpatch_root: Option<String>,
}

#[cfg(feature = "stdio")]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct SuggestArgs {
    repo_root: String,
    file: String,
    lemma: String,
    #[serde(default)]
    timeout_s: Option<u64>,
    // Schema compatibility only (unused).
    #[serde(default)]
    proofpatch_root: Option<String>,
}

#[cfg(feature = "stdio")]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct RubberduckArgs {
    repo_root: String,
    file: String,
    lemma: String,
    #[serde(default)]
    diagnostics: Option<String>,
    // Schema compatibility only (unused).
    #[serde(default)]
    proofpatch_root: Option<String>,
}

#[cfg(feature = "stdio")]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct LoopArgs {
    repo_root: String,
    file: String,
    lemma: String,
    #[serde(default)]
    max_iters: Option<u64>,
    #[serde(default)]
    timeout_s: Option<u64>,
    // Schema compatibility only (unused).
    #[serde(default)]
    proofpatch_root: Option<String>,
}

#[cfg(feature = "stdio")]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct ReportHtmlArgs {
    repo_root: String,
    files: Vec<String>,
    #[serde(default)]
    timeout_s: Option<u64>,
    #[serde(default)]
    max_sorries: Option<u64>,
    #[serde(default)]
    context_lines: Option<u64>,
    #[serde(default)]
    include_raw_verify: Option<bool>,
    #[serde(default)]
    output_path: Option<String>,
}

#[cfg(feature = "stdio")]
#[derive(Clone)]
struct ProofpatchStdioMcp {
    tool_router: RmcpToolRouter<Self>,
}

#[cfg(feature = "stdio")]
impl ProofpatchStdioMcp {
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }
}

#[cfg(feature = "stdio")]
#[tool_router]
impl ProofpatchStdioMcp {
    #[tool(description = "Triage a file: verify_summary + locate_sorries")]
    async fn proofpatch_triage_file(
        &self,
        params: Parameters<RepoFileSorriesArgs>,
    ) -> Result<CallToolResult, McpError> {
        let repo_root = PathBuf::from(&params.0.repo_root);
        let file = params.0.file.clone();
        let timeout_s = params.0.timeout_s.unwrap_or(180);
        let max_sorries = params.0.max_sorries.unwrap_or(50) as usize;
        let context_lines = params.0.context_lines.unwrap_or(1) as usize;
        let include_raw_verify = params.0.include_raw_verify.unwrap_or(false);
        let include_context_pack = params.0.include_context_pack.unwrap_or(true);
        let pack_context_lines = params.0.pack_context_lines.unwrap_or(6) as usize;
        let pack_nearby_lines = params.0.pack_nearby_lines.unwrap_or(60) as usize;
        let pack_max_nearby_decls = params.0.pack_max_nearby_decls.unwrap_or(20) as usize;
        let pack_max_imports = params.0.pack_max_imports.unwrap_or(20) as usize;
        let include_prompts = params.0.include_prompts.unwrap_or(true);
        let output_path = params.0.output_path.clone();

        let repo_root = resolve_lean_repo_root(repo_root, Some(&file))
            .map_err(|e| McpError::invalid_params(e, None))?;
        plc::load_dotenv_smart(&repo_root);

        let raw = plc::verify_lean_file(&repo_root, &file, StdDuration::from_secs(timeout_s))
            .await
            .map_err(|e| McpError::internal_error(e, None))?;
        let raw_v = serde_json::to_value(raw).map_err(|e| {
            McpError::internal_error(format!("failed to serialize verify result: {e}"), None)
        })?;
        let summary = summarize_verify_like_output(&raw_v);
        let locs = plc::locate_sorries_in_file(&repo_root, &file, max_sorries, context_lines)
            .map_err(|e| McpError::internal_error(e, None))?;

        let nearest = summary
            .get("first_error_loc")
            .and_then(|v| v.get("line"))
            .and_then(|v| v.as_u64())
            .map(|l| l as i64)
            .and_then(|err_line| {
                locs.iter()
                    .min_by_key(|s| (s.line as i64 - err_line).abs())
                    .cloned()
            });

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
                        pack_max_nearby_decls,
                        pack_max_imports,
                    )
                    .ok()
                })
                .and_then(|p| serde_json::to_value(p).ok());

            let pack_nearest = nearest
                .as_ref()
                .and_then(|s| {
                    plc::build_context_pack(
                        &repo_root,
                        &file,
                        None,
                        Some(s.line),
                        pack_context_lines,
                        pack_nearby_lines,
                        pack_max_nearby_decls,
                        pack_max_imports,
                    )
                    .ok()
                })
                .and_then(|p| serde_json::to_value(p).ok());

            (pack_first, pack_nearest)
        } else {
            (None, None)
        };

        let first_error_text = summary
            .get("first_error")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let rubberduck_prompt_first_error = if include_prompts {
            context_pack_first_error
                .as_ref()
                .and_then(|v| v.get("focus"))
                .and_then(|f| f.get("excerpt"))
                .and_then(|e| e.as_str())
                .and_then(|excerpt| {
                    let label = summary
                        .get("first_error_loc")
                        .and_then(|v| v.get("line"))
                        .and_then(|v| v.as_u64())
                        .map(|l| format!("line:{l}"))
                        .unwrap_or_else(|| "line:?".to_string());
                    plc::build_rubberduck_prompt_from_excerpt(
                        &repo_root,
                        &file,
                        &label,
                        excerpt,
                        first_error_text.as_deref(),
                    )
                    .ok()
                })
                .and_then(|p| serde_json::to_value(p).ok())
        } else {
            None
        };

        let patch_prompt_nearest_sorry = if include_prompts {
            nearest
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
            let first_error_line = summary
                .get("first_error_loc")
                .and_then(|v| v.get("line"))
                .and_then(|v| v.as_u64());

            if has_error {
                serde_json::json!({
                    "kind": "fix_first_error",
                    "prompt": rubberduck_prompt_first_error,
                    "line": first_error_line,
                })
            } else if nearest.is_some() {
                serde_json::json!({
                    "kind": "patch_nearest_sorry",
                    "prompt": patch_prompt_nearest_sorry,
                    "region": nearest.as_ref().map(|s| serde_json::json!({"start_line": s.region_start, "end_line": s.region_end})),
                })
            } else {
                serde_json::json!({ "kind": "noop" })
            }
        };

        let full = serde_json::json!({
            "repo_root": repo_root.display().to_string(),
            "file": file,
            "verify": if include_raw_verify {
                serde_json::json!({ "summary": summary, "raw": raw_v })
            } else {
                serde_json::json!({ "summary": summary, "raw": serde_json::Value::Null })
            },
            "sorries": { "count": locs.len(), "locations": locs },
            "nearest_sorry_to_first_error": nearest,
            "context_pack_first_error": context_pack_first_error,
            "context_pack_nearest_sorry": context_pack_nearest_sorry,
            "rubberduck_prompt_first_error": rubberduck_prompt_first_error,
            "patch_prompt_nearest_sorry": patch_prompt_nearest_sorry,
            "next_action": next_action,
        });

        if let Some(p) = output_path {
            let path = std::path::PathBuf::from(&p);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    McpError::internal_error(
                        format!("failed to create output dir {}: {}", parent.display(), e),
                        None,
                    )
                })?;
            }
            let s = serde_json::to_string_pretty(&full).map_err(|e| {
                McpError::internal_error(format!("failed to encode json: {e}"), None)
            })?;
            std::fs::write(&path, s.as_bytes()).map_err(|e| {
                McpError::internal_error(format!("failed to write {}: {e}", path.display()), None)
            })?;

            let small = serde_json::json!({
                "ok": true,
                "written_path": path.display().to_string(),
                "repo_root": full.get("repo_root").cloned().unwrap_or(serde_json::Value::Null),
                "file": full.get("file").cloned().unwrap_or(serde_json::Value::Null),
                "sorries": locs.len(),
                "next_action": full.get("next_action").cloned().unwrap_or(serde_json::Value::Null),
            });
            return Ok(CallToolResult::success(vec![Content::text(
                small.to_string(),
            )]));
        }

        Ok(CallToolResult::success(vec![Content::text(
            full.to_string(),
        )]))
    }

    // ---------------------------------------------------------------------
    // Thin stdio wrappers for the rest of the proofpatch tool surface.
    //
    // These delegate to the existing `axum-mcp` Tool implementations so HTTP
    // and stdio stay behaviorally consistent.
    //
    // Some endpoints use typed `JsonSchema` arg structs (better UX in Cursor); others keep
    // `serde_json::Value` to avoid duplicating schemas for less frequently used tools.
    // ---------------------------------------------------------------------

    #[tool(description = "Extract the (system,user) prompt + excerpt for a lemma (`proofpatch prompt`).")]
    async fn proofpatch_prompt(
        &self,
        params: Parameters<PromptArgs>,
    ) -> Result<CallToolResult, McpError> {
        let tool = ProofpatchPromptTool;
        let v = serde_json::to_value(&params.0)
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;
        let out = tool
            .call(&v)
            .await
            .map_err(|e| McpError::invalid_params(e, None))?;
        Ok(CallToolResult::success(vec![Content::text(out.to_string())]))
    }

    #[tool(description = "Elaboration-check a file (`proofpatch verify`).")]
    async fn proofpatch_verify(
        &self,
        params: Parameters<VerifyArgs>,
    ) -> Result<CallToolResult, McpError> {
        let tool = ProofpatchVerifyTool;
        let v = serde_json::to_value(&params.0)
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;
        let out = tool
            .call(&v)
            .await
            .map_err(|e| McpError::invalid_params(e, None))?;
        Ok(CallToolResult::success(vec![Content::text(out.to_string())]))
    }

    #[tool(description = "Elaboration-check a file, returning a small summary plus raw output (`proofpatch verify`).")]
    async fn proofpatch_verify_summary(
        &self,
        params: Parameters<VerifyArgs>,
    ) -> Result<CallToolResult, McpError> {
        let tool = ProofpatchVerifySummaryTool;
        let v = serde_json::to_value(&params.0)
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;
        let out = tool
            .call(&v)
            .await
            .map_err(|e| McpError::invalid_params(e, None))?;
        Ok(CallToolResult::success(vec![Content::text(out.to_string())]))
    }

    #[tool(description = "Locate `sorry` tokens in a file with line/col and suggested patch regions.")]
    async fn proofpatch_locate_sorries(
        &self,
        params: Parameters<LocateSorriesArgs>,
    ) -> Result<CallToolResult, McpError> {
        let tool = ProofpatchLocateSorriesTool;
        let v = serde_json::to_value(&params.0)
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;
        let out = tool
            .call(&v)
            .await
            .map_err(|e| McpError::invalid_params(e, None))?;
        Ok(CallToolResult::success(vec![Content::text(out.to_string())]))
    }

    #[tool(description = "Suggest a proof by running the configured LLM router (`proofpatch suggest`).")]
    async fn proofpatch_suggest(
        &self,
        params: Parameters<SuggestArgs>,
    ) -> Result<CallToolResult, McpError> {
        let tool = ProofpatchSuggestTool;
        let v = serde_json::to_value(&params.0)
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;
        let out = tool
            .call(&v)
            .await
            .map_err(|e| McpError::internal_error(e, None))?;
        Ok(CallToolResult::success(vec![Content::text(out.to_string())]))
    }

    #[tool(description = "Patch a lemma’s first `sorry` with provided Lean code, then verify (`proofpatch patch`).")]
    async fn proofpatch_patch(
        &self,
        params: Parameters<PatchArgs>,
    ) -> Result<CallToolResult, McpError> {
        let tool = ProofpatchPatchTool;
        let v = serde_json::to_value(&params.0)
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;
        let out = tool
            .call(&v)
            .await
            .map_err(|e| McpError::internal_error(e, None))?;
        Ok(CallToolResult::success(vec![Content::text(out.to_string())]))
    }

    #[tool(description = "Patch the first `sorry` within a (line-based) region and verify.")]
    async fn proofpatch_patch_region(
        &self,
        params: Parameters<PatchRegionArgs>,
    ) -> Result<CallToolResult, McpError> {
        let tool = ProofpatchPatchRegionTool;
        let v = serde_json::to_value(&params.0)
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;
        let out = tool
            .call(&v)
            .await
            .map_err(|e| McpError::internal_error(e, None))?;
        Ok(CallToolResult::success(vec![Content::text(out.to_string())]))
    }

    #[tool(description = "Build a rubberduck/ideation prompt for a lemma (no proof code; plan + next moves).")]
    async fn proofpatch_rubberduck_prompt(
        &self,
        params: Parameters<RubberduckArgs>,
    ) -> Result<CallToolResult, McpError> {
        let tool = ProofpatchRubberduckPromptTool;
        let v = serde_json::to_value(&params.0)
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;
        let out = tool
            .call(&v)
            .await
            .map_err(|e| McpError::invalid_params(e, None))?;
        Ok(CallToolResult::success(vec![Content::text(out.to_string())]))
    }

    #[tool(description = "Bounded loop: suggest → patch first `sorry` in lemma → verify (`proofpatch loop`).")]
    async fn proofpatch_loop(
        &self,
        params: Parameters<LoopArgs>,
    ) -> Result<CallToolResult, McpError> {
        let tool = ProofpatchLoopTool;
        let v = serde_json::to_value(&params.0)
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;
        let out = tool
            .call(&v)
            .await
            .map_err(|e| McpError::internal_error(e, None))?;
        Ok(CallToolResult::success(vec![Content::text(out.to_string())]))
    }

    #[tool(description = "Execute one safe agent step (no LLM): verify → mechanical fix → verify")]
    async fn proofpatch_agent_step(
        &self,
        params: Parameters<AgentStepArgs>,
    ) -> Result<CallToolResult, McpError> {
        let repo_root = PathBuf::from(&params.0.repo_root);
        let file = params.0.file.clone();
        let timeout_s = params.0.timeout_s.unwrap_or(180);
        let write = params.0.write.unwrap_or(false);
        let output_path = params.0.output_path.clone();

        let repo_root = resolve_lean_repo_root(repo_root, Some(&file))
            .map_err(|e| McpError::invalid_params(e, None))?;
        plc::load_dotenv_smart(&repo_root);

        let abs = repo_root.join(&file);
        if !abs.exists() {
            return Err(McpError::invalid_params(
                format!("File not found: {}", abs.display()),
                None,
            ));
        }
        let original_text = std::fs::read_to_string(&abs)
            .map_err(|e| McpError::internal_error(format!("read {}: {e}", abs.display()), None))?;

        let verify0 = plc::verify_lean_file(&repo_root, &file, StdDuration::from_secs(timeout_s))
            .await
            .map_err(|e| McpError::internal_error(e, None))?;
        let first_error_loc = plc::parse_first_error_loc(&verify0.stdout, &verify0.stderr);
        let first_error_text = verify0
            .stdout
            .lines()
            .find(|l| l.contains(": error:"))
            .map(|s| s.to_string());

        let (patched_text, edits) = apply_mechanical_fixes_for_first_error(
            &original_text,
            first_error_loc.as_ref().map(|l| l.line),
            first_error_text.as_deref(),
        );

        let wrote_file = if write && !edits.is_empty() {
            std::fs::write(&abs, patched_text.as_bytes()).map_err(|e| {
                McpError::internal_error(format!("write {}: {e}", abs.display()), None)
            })?;
            Some(abs.display().to_string())
        } else {
            None
        };

        let verify1 = if write && !edits.is_empty() {
            plc::verify_lean_file(&repo_root, &file, StdDuration::from_secs(timeout_s))
                .await
                .map_err(|e| McpError::internal_error(e, None))?
        } else {
            plc::verify_lean_text(&repo_root, &patched_text, StdDuration::from_secs(timeout_s))
                .await
                .map_err(|e| McpError::internal_error(e, None))?
        };

        let dag = json!({
            "nodes": [
                { "id": "verify0", "kind": "verify", "ok": verify0.ok, "returncode": verify0.returncode, "first_error_loc": first_error_loc, "first_error": first_error_text },
                { "id": "mech_fix1", "kind": "mechanical_fix", "applied": !edits.is_empty(), "edits": edits, "write": write },
                { "id": "verify1", "kind": "verify", "ok": verify1.ok, "returncode": verify1.returncode, "first_error_loc": plc::parse_first_error_loc(&verify1.stdout, &verify1.stderr), "first_error": verify1.stdout.lines().find(|l| l.contains(": error:")) }
            ],
            "edges": [
                { "from": "verify0", "to": "mech_fix1" },
                { "from": "mech_fix1", "to": "verify1" }
            ]
        });

        let full = json!({
            "ok": true,
            "repo_root": repo_root.display().to_string(),
            "file": file,
            "written_file": wrote_file,
            "dag": dag
        });

        if let Some(p) = output_path {
            let path = std::path::PathBuf::from(&p);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    McpError::internal_error(
                        format!("failed to create output dir {}: {}", parent.display(), e),
                        None,
                    )
                })?;
            }
            let s = serde_json::to_string_pretty(&full).map_err(|e| {
                McpError::internal_error(format!("failed to encode json: {}", e), None)
            })?;
            std::fs::write(&path, s.as_bytes()).map_err(|e| {
                McpError::internal_error(format!("failed to write {}: {}", path.display(), e), None)
            })?;

            let small = json!({
                "ok": true,
                "written_path": path.display().to_string(),
                "file": params.0.file,
                "written_file": wrote_file,
                "verify1_ok": verify1.ok
            });
            return Ok(CallToolResult::success(vec![Content::text(
                small.to_string(),
            )]));
        }

        Ok(CallToolResult::success(vec![Content::text(
            full.to_string(),
        )]))
    }

    #[tool(description = "Build a JSON-first context pack for file + decl/line")]
    async fn proofpatch_context_pack(
        &self,
        params: Parameters<ContextPackArgs>,
    ) -> Result<CallToolResult, McpError> {
        let repo_root = PathBuf::from(&params.0.repo_root);
        let file = params.0.file.clone();
        let decl = params.0.decl.clone();
        let line = params.0.line.map(|x| x as usize);
        let context_lines = params.0.context_lines.unwrap_or(25) as usize;
        let nearby_lines = params.0.nearby_lines.unwrap_or(80) as usize;
        let max_nearby_decls = params.0.max_nearby_decls.unwrap_or(30) as usize;
        let max_imports = params.0.max_imports.unwrap_or(50) as usize;

        let repo_root = resolve_lean_repo_root(repo_root, Some(&file))
            .map_err(|e| McpError::invalid_params(e, None))?;
        let pack = plc::build_context_pack(
            &repo_root,
            &file,
            decl.as_deref(),
            line,
            context_lines,
            nearby_lines,
            max_nearby_decls,
            max_imports,
        )
        .map_err(|e| McpError::internal_error(e, None))?;

        let out = serde_json::to_value(pack).map_err(|e| {
            McpError::internal_error(format!("failed to serialize context pack: {e}"), None)
        })?;
        Ok(CallToolResult::success(vec![Content::text(
            out.to_string(),
        )]))
    }

    #[tool(description = "Triage many files and write a small HTML report")]
    async fn proofpatch_report_html(
        &self,
        params: Parameters<ReportHtmlArgs>,
    ) -> Result<CallToolResult, McpError> {
        // Delegate to the existing Tool implementation to avoid duplicating report logic.
        let tool = ProofpatchReportHtmlTool;
        let v = serde_json::to_value(&params.0)
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;
        let out = tool
            .call(&v)
            .await
            .map_err(|e| McpError::internal_error(e, None))?;
        Ok(CallToolResult::success(vec![Content::text(
            out.to_string(),
        )]))
    }
}

#[cfg(feature = "stdio")]
#[tool_handler]
impl rmcp::ServerHandler for ProofpatchStdioMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            server_info: Implementation {
                // What clients should display as the server name.
                // (Default is the rmcp framework name, which is misleading for debugging.)
                name: "proofpatch-mcp".to_string(),
                title: Some("proofpatch-mcp".to_string()),
                version: env!("CARGO_PKG_VERSION").to_string(),
                icons: None,
                website_url: None,
            },
            instructions: Some(
                "Tools for Lean proof triage/patching loops (proofpatch). JSON-only, stdout reserved for MCP frames."
                    .to_string(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    // Don’t emit logs to stdout if this ever becomes stdio-based.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Minimal CLI:
    // - `proofpatch-mcp --help`     => print usage and exit
    // - `proofpatch-mcp --version`  => print version and exit
    // - `proofpatch-mcp mcp-stdio`  => stdio MCP server (Cursor can spawn it)
    // - default                     => HTTP MCP server (daemon)
    let arg1 = std::env::args().nth(1);
    if matches!(arg1.as_deref(), Some("-h" | "--help" | "help")) {
        println!("proofpatch-mcp");
        println!("");
        println!("Usage:");
        println!("  proofpatch-mcp            # HTTP MCP server (default)");
        println!("  proofpatch-mcp mcp-stdio  # stdio MCP server (for Cursor)");
        println!("");
        println!("Env:");
        println!("  PROOFPATCH_MCP_ADDR=127.0.0.1:8087");
        println!("  PROOFPATCH_MCP_TOOL_TIMEOUT_S=180");
        println!("  PROOFPATCH_MCP_TOOLSET=minimal|full");
        return Ok(());
    }
    if matches!(arg1.as_deref(), Some("-V" | "--version" | "version")) {
        println!("proofpatch-mcp {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    if arg1.as_deref() == Some("mcp-stdio") {
        #[cfg(feature = "stdio")]
        {
            let service = ProofpatchStdioMcp::new();
            let running = service
                .serve(stdio())
                .await
                .map_err(|e| format!("failed to start stdio MCP server: {e:?}"))?;
            running
                .waiting()
                .await
                .map_err(|e| format!("stdio MCP server task join failed: {e:?}"))?;
            return Ok(());
        }
        #[cfg(not(feature = "stdio"))]
        {
            return Err("mcp-stdio requires compile-time feature `stdio`".into());
        }
    }

    let addr = std::env::var("PROOFPATCH_MCP_ADDR").unwrap_or_else(|_| "127.0.0.1:8087".to_string());

    // The default axum-mcp tool timeout is 30s, but `lake build` / `lake env lean` can take longer
    // even on successful runs (big mathlib files + linters + first-time dependency builds).
    //
    // Override with: PROOFPATCH_MCP_TOOL_TIMEOUT_S=900 (or larger)
    let tool_timeout_s: u64 = std::env::var("PROOFPATCH_MCP_TOOL_TIMEOUT_S")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(180);
    let config = ServerConfig::new().with_tool_timeout(StdDuration::from_secs(tool_timeout_s));

    // Tool surface: default to a small "minimal" set for agent ergonomics. Opt into `full`
    // for the complete tool surface.
    //
    // - minimal (default): verify/triage/patch + tree-search
    // - full: everything (legacy)
    let toolset = std::env::var("PROOFPATCH_MCP_TOOLSET")
        .unwrap_or_else(|_| "minimal".to_string())
        .trim()
        .to_lowercase();

    let server = match toolset.as_str() {
        "minimal" | "" => McpServer::with_config(config)
            .tool("proofpatch_verify_summary", ProofpatchVerifySummaryTool)?
            .tool("proofpatch_locate_sorries", ProofpatchLocateSorriesTool)?
            .tool("proofpatch_triage_file", ProofpatchTriageFileTool)?
            .tool("proofpatch_patch_region", ProofpatchPatchRegionTool)?
            .tool("proofpatch_patch_nearest", ProofpatchPatchNearestTool)?
            .tool("proofpatch_tree_search_nearest", ProofpatchTreeSearchNearestTool)?,
        "full" => McpServer::with_config(config)
            .tool("proofpatch_prompt", ProofpatchPromptTool)?
            .tool("proofpatch_verify", ProofpatchVerifyTool)?
            .tool("proofpatch_verify_summary", ProofpatchVerifySummaryTool)?
            .tool("proofpatch_suggest", ProofpatchSuggestTool)?
            .tool("proofpatch_patch", ProofpatchPatchTool)?
            .tool("proofpatch_patch_region", ProofpatchPatchRegionTool)?
            .tool("proofpatch_patch_nearest", ProofpatchPatchNearestTool)?
            .tool("proofpatch_tree_search_nearest", ProofpatchTreeSearchNearestTool)?
            .tool("proofpatch_locate_sorries", ProofpatchLocateSorriesTool)?
            .tool("proofpatch_context_pack", ProofpatchContextPackTool)?
            .tool("proofpatch_triage_file", ProofpatchTriageFileTool)?
            .tool("proofpatch_agent_step", ProofpatchAgentStepTool)?
            .tool("proofpatch_report_html", ProofpatchReportHtmlTool)?
            .tool("proofpatch_rubberduck_prompt", ProofpatchRubberduckPromptTool)?
            .tool("proofpatch_loop", ProofpatchLoopTool)?,
        other => {
            return Err(format!(
                "unknown PROOFPATCH_MCP_TOOLSET={other} (expected minimal|full)"
            )
            .into())
        }
    };

    eprintln!("proofpatch MCP server listening on http://{addr}");
    server.serve(&addr).await?;
    Ok(())
}

#[tokio::main]
#[allow(dead_code)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    run().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn mk(p: &Path) {
        std::fs::create_dir_all(p).expect("mkdir");
    }

    fn touch(p: &Path, content: &str) {
        if let Some(parent) = p.parent() {
            mk(parent);
        }
        std::fs::write(p, content.as_bytes()).expect("write");
    }

    #[test]
    fn resolve_lean_repo_root_can_anchor_on_file_inside_subdir_repo() {
        let base = std::env::temp_dir().join(format!("proofpatch-test-{}", uuid::Uuid::new_v4()));
        let ws_root = base.join("ws");
        let lean_root = ws_root.join("leanproj");
        let file_abs = lean_root.join("Some").join("File.lean");

        mk(&ws_root);
        mk(&lean_root);
        touch(&lean_root.join("lean-toolchain"), "leanprover/lean4:stable\n");
        touch(&lean_root.join("lakefile.toml"), "[package]\nname = \"leanproj\"\n");
        touch(&file_abs, "theorem hello : True := by\n  trivial\n");

        // User passes a super-workspace root, but a file path that points into a Lean project.
        let resolved = resolve_lean_repo_root(ws_root.clone(), Some("leanproj/Some/File.lean"))
            .expect("expected lean repo root");
        assert_eq!(
            resolved.canonicalize().unwrap(),
            lean_root.canonicalize().unwrap()
        );

        // Same, but with an absolute file path.
        let resolved2 = resolve_lean_repo_root(ws_root, Some(file_abs.to_string_lossy().as_ref()))
            .expect("expected lean repo root");
        assert_eq!(
            resolved2.canonicalize().unwrap(),
            lean_root.canonicalize().unwrap()
        );
    }

    #[test]
    fn mechanical_fix_inserts_classical_before_decide_on_missing_decidable() {
        let txt = "theorem t : True := by\n  have h : True := by\n    decide\n  exact True.intro\n";
        let msg = "Foo.lean:3:5: error: failed to synthesize\n  Decidable\n    True\n";
        let (out, edits) = apply_mechanical_fixes_for_first_error(txt, Some(3), Some(msg));
        assert!(out.contains("\n    classical\n    decide\n"));
        assert_eq!(edits.len(), 1);
        assert_eq!(
            edits[0].get("kind").and_then(|v| v.as_str()).unwrap_or(""),
            "insert_classical_before_decide"
        );
    }
}
