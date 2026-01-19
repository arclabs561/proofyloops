//! proofloops MCP server (HTTP via axum-mcp).
//!
//! Exposes `proofloops-core` as MCP tools over HTTP/stdio.
//!
//! Run:
//! ```bash
//! cd /Users/arc/Documents/dev/proofloops/mcp-server
//! cargo run
//! ```
//!
//! Then:
//! - `curl http://127.0.0.1:8087/health`
//! - `curl http://127.0.0.1:8087/tools/list`
//! - `curl -X POST http://127.0.0.1:8087/tools/call -H 'Content-Type: application/json' -d '{"name":"proofloops_prompt","arguments":{"repo_root":"/Users/arc/Documents/dev/geometry-of-numbers","file":"Covolume/Legendre/Ankeny.lean","lemma":"ankeny_even_padicValNat_of_mem_primeFactors"}}'`
//!
//! Configuration:
//! - `PROOFLOOPS_MCP_ADDR` (default: `127.0.0.1:8087`) (legacy: `PROOFYLOOPS_MCP_ADDR`)
//! - `PROOFLOOPS_MCP_TOOL_TIMEOUT_S` (default: `180`) (legacy: `PROOFYLOOPS_MCP_TOOL_TIMEOUT_S`)

use async_trait::async_trait;
use axum_mcp::{
    extract_integer_opt, extract_string, extract_string_opt, McpServer, ServerConfig, Tool,
};
use proofloops_core as plc;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::Duration as StdDuration;

// Optional stdio MCP transport (Cursor can spawn without a daemon).
#[cfg(feature = "stdio")]
use rmcp::{
    handler::server::router::tool::ToolRouter as RmcpToolRouter,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
    transport::stdio,
    ErrorData as McpError, ServiceExt,
};
#[cfg(feature = "stdio")]
use schemars::JsonSchema;
#[cfg(feature = "stdio")]
use serde::Deserialize;

fn default_proofloops_root() -> PathBuf {
    // `.../proofloops/mcp-server` → `.../proofloops`
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("mcp-server should be nested under proofloops/")
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
        if line.contains(": error:") || line.starts_with("error:") {
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
    let (kind, idx) = if let Some(i) = line.find(": error:") {
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

    let warning_count = count_substring(stdout, ": warning:")
        + count_substring(stderr, ": warning:")
        + count_substring(stderr, "warning:");
    let error_count = count_substring(stdout, ": error:")
        + count_substring(stderr, ": error:")
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
            "warnings": warning_count
        },
        "first_error": first_error,
        "first_error_loc": first_error_loc,
    })
}

// NOTE: The MCP server used to bridge to a Python CLI.
// It is Rust-native now (proofloops-core has the provider router), so there is no reason to shell out.

fn proofloops_root_from_args(args: &Value) -> Result<PathBuf, String> {
    if let Ok(env_root) = std::env::var("PROOFLOOPS_ROOT") {
        if !env_root.trim().is_empty() {
            return Ok(PathBuf::from(env_root));
        }
    }
    if let Ok(env_root) = std::env::var("PROOFYLOOPS_ROOT") {
        if !env_root.trim().is_empty() {
            return Ok(PathBuf::from(env_root));
        }
    }
    let root = extract_string_opt(args, "proofloops_root")
        .or_else(|| extract_string_opt(args, "proofyloops_root"))
        .unwrap_or_else(|| default_proofloops_root().to_string_lossy().to_string());
    Ok(PathBuf::from(root))
}

fn repo_root_from_args(args: &Value) -> Result<PathBuf, String> {
    let repo_root = PathBuf::from(extract_string(args, "repo_root")?);
    // Parse `proofloops_root` / `proofyloops_root` (legacy) to keep schemas honest,
    // but do not use it for anything. Repo-root resolution is independent of helper-root.
    let _ = proofloops_root_from_args(args)?;
    Ok(repo_root)
}

struct ProofyloopsPromptTool;

#[async_trait]
impl Tool for ProofyloopsPromptTool {
    fn description(&self) -> &str {
        "Extract the (system,user) prompt + excerpt for a lemma (`proofloops prompt`)."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "repo_root": { "type": "string", "description": "Lean repo root" },
                "file": { "type": "string", "description": "File path relative to repo root" },
                "lemma": { "type": "string", "description": "Lemma name to extract" },
                "timeout_s": { "type": "integer", "default": 30 },
                "proofloops_root": { "type": "string", "description": "Path to proofloops (defaults to sibling of this crate)" },
                "proofyloops_root": { "type": "string", "description": "Legacy alias for proofloops_root" }
            },
            "required": ["repo_root", "file", "lemma"]
        })
    }

    async fn call(&self, args: &Value) -> Result<Value, String> {
        let repo_root = repo_root_from_args(args)?;
        let file = extract_string(args, "file")?;
        let lemma = extract_string(args, "lemma")?;
        // Keep `proofyloops_root`/`timeout_s` in the schema for backward compatibility,
        // but the core prompt logic is now Rust-native.
        let _ = extract_u64_opt(args, "timeout_s")?.unwrap_or(30);
        let payload = plc::build_proof_prompt(&repo_root, &file, &lemma)?;
        serde_json::to_value(payload).map_err(|e| format!("failed to serialize payload: {}", e))
    }
}

struct ProofyloopsVerifyTool;

#[async_trait]
impl Tool for ProofyloopsVerifyTool {
    fn description(&self) -> &str {
        "Elaboration-check a file (`proofloops verify`)."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "repo_root": { "type": "string" },
                "file": { "type": "string", "description": "File path relative to repo root" },
                "timeout_s": { "type": "integer", "default": 120 },
                "proofloops_root": { "type": "string" },
                "proofyloops_root": { "type": "string", "description": "Legacy alias for proofloops_root" }
            },
            "required": ["repo_root", "file"]
        })
    }

    async fn call(&self, args: &Value) -> Result<Value, String> {
        let repo_root = repo_root_from_args(args)?;
        let file = extract_string(args, "file")?;
        let timeout_s = extract_u64_opt(args, "timeout_s")?.unwrap_or(120);
        let raw =
            plc::verify_lean_file(&repo_root, &file, StdDuration::from_secs(timeout_s)).await?;
        serde_json::to_value(raw).map_err(|e| format!("failed to serialize verify result: {}", e))
    }
}

struct ProofyloopsVerifySummaryTool;

#[async_trait]
impl Tool for ProofyloopsVerifySummaryTool {
    fn description(&self) -> &str {
        "Elaboration-check a file, returning a small summary plus raw output (`proofloops verify`)."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "repo_root": { "type": "string" },
                "file": { "type": "string", "description": "File path relative to repo root" },
                "timeout_s": { "type": "integer", "default": 120 },
                "proofloops_root": { "type": "string" },
                "proofyloops_root": { "type": "string", "description": "Legacy alias for proofloops_root" }
            },
            "required": ["repo_root", "file"]
        })
    }

    async fn call(&self, args: &Value) -> Result<Value, String> {
        let repo_root = repo_root_from_args(args)?;
        let file = extract_string(args, "file")?;
        let timeout_s = extract_u64_opt(args, "timeout_s")?.unwrap_or(120);
        let raw =
            plc::verify_lean_file(&repo_root, &file, StdDuration::from_secs(timeout_s)).await?;
        let raw_v = serde_json::to_value(raw)
            .map_err(|e| format!("failed to serialize verify result: {}", e))?;
        Ok(json!({"summary": summarize_verify_like_output(&raw_v), "raw": raw_v}))
    }
}

struct ProofyloopsSuggestTool;

#[async_trait]
impl Tool for ProofyloopsSuggestTool {
    fn description(&self) -> &str {
        "Suggest a proof by running the configured LLM router (`proofloops suggest`)."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "repo_root": { "type": "string" },
                "file": { "type": "string" },
                "lemma": { "type": "string" },
                "timeout_s": { "type": "integer", "default": 120 },
                "proofloops_root": { "type": "string" },
                "proofyloops_root": { "type": "string", "description": "Legacy alias for proofloops_root" }
            },
            "required": ["repo_root", "file", "lemma"]
        })
    }

    async fn call(&self, args: &Value) -> Result<Value, String> {
        let repo_root = repo_root_from_args(args)?;
        let file = extract_string(args, "file")?;
        let lemma = extract_string(args, "lemma")?;
        let timeout_s = extract_u64_opt(args, "timeout_s")?.unwrap_or(120);
        // Keep `proofyloops_root` in the schema for backward compatibility,
        // but do not shell out; use Rust core LLM router instead.
        let _ = proofloops_root_from_args(args)?;

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

struct ProofyloopsPatchTool;

#[async_trait]
impl Tool for ProofyloopsPatchTool {
    fn description(&self) -> &str {
        "Patch a lemma’s first `sorry` with provided Lean code, then verify (`proofloops patch`)."
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
                "proofloops_root": { "type": "string" },
                "proofyloops_root": { "type": "string", "description": "Legacy alias for proofloops_root" }
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

        let repo_root = plc::find_lean_repo_root(&repo_root)?;
        plc::load_dotenv_smart(&repo_root);

        let p = repo_root.join(&file);
        if !p.exists() {
            return Err(format!("File not found: {}", p.display()));
        }
        let original = std::fs::read_to_string(&p)
            .map_err(|e| format!("failed to read {}: {}", p.display(), e))?;

        let patched = plc::patch_first_sorry_in_decl(&original, &lemma, &replacement)?;
        let still_has_sorry = plc::decl_block_contains_sorry(&patched.text, &lemma)?;
        let verify =
            plc::verify_lean_text(&repo_root, &patched.text, StdDuration::from_secs(timeout_s))
                .await?;

        Ok(json!({
            "file": p.display().to_string(),
            "lemma": lemma,
            "patch": {
                "line": patched.line,
                "before": patched.before,
                "after": patched.after,
                "indent": patched.indent,
            },
            "lemma_still_contains_sorry": still_has_sorry,
            "verify": verify,
        }))
    }
}

struct ProofyloopsPatchRegionTool;

#[async_trait]
impl Tool for ProofyloopsPatchRegionTool {
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
        let timeout_s = extract_u64_opt(args, "timeout_s")?.unwrap_or(120);

        let repo_root = plc::find_lean_repo_root(&repo_root)?;
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

        let verify =
            plc::verify_lean_text(&repo_root, &patched.text, StdDuration::from_secs(timeout_s))
                .await?;

        Ok(json!({
            "file": p.display().to_string(),
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
            "verify": verify,
        }))
    }
}

struct ProofyloopsLocateSorriesTool;

#[async_trait]
impl Tool for ProofyloopsLocateSorriesTool {
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
        let repo_root = repo_root_from_args(args)?;
        let file = extract_string(args, "file")?;
        let max_results = extract_u64_opt(args, "max_results")?.unwrap_or(50) as usize;
        let context_lines = extract_u64_opt(args, "context_lines")?.unwrap_or(2) as usize;

        let repo_root = plc::find_lean_repo_root(&repo_root)?;
        let locs = plc::locate_sorries_in_file(&repo_root, &file, max_results, context_lines)?;

        Ok(json!({
            "repo_root": repo_root.display().to_string(),
            "file": file,
            "count": locs.len(),
            "locations": locs,
        }))
    }
}

struct ProofyloopsContextPackTool;

#[async_trait]
impl Tool for ProofyloopsContextPackTool {
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
        let repo_root = repo_root_from_args(args)?;
        let file = extract_string(args, "file")?;
        let decl = extract_string_opt(args, "decl");
        let line = extract_u64_opt(args, "line")?.map(|x| x as usize);
        let context_lines = extract_u64_opt(args, "context_lines")?.unwrap_or(25) as usize;
        let nearby_lines = extract_u64_opt(args, "nearby_lines")?.unwrap_or(80) as usize;
        let max_nearby_decls = extract_u64_opt(args, "max_nearby_decls")?.unwrap_or(30) as usize;
        let max_imports = extract_u64_opt(args, "max_imports")?.unwrap_or(50) as usize;

        let repo_root = plc::find_lean_repo_root(&repo_root)?;
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

struct ProofyloopsTriageFileTool;

#[async_trait]
impl Tool for ProofyloopsTriageFileTool {
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
        let repo_root = repo_root_from_args(args)?;
        let file = extract_string(args, "file")?;
        let timeout_s = extract_u64_opt(args, "timeout_s")?.unwrap_or(180);
        let max_sorries = extract_u64_opt(args, "max_sorries")?.unwrap_or(50) as usize;
        let context_lines = extract_u64_opt(args, "context_lines")?.unwrap_or(1) as usize;
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

        let repo_root = plc::find_lean_repo_root(&repo_root)?;
        plc::load_dotenv_smart(&repo_root);

        let raw =
            plc::verify_lean_file(&repo_root, &file, StdDuration::from_secs(timeout_s)).await?;
        let raw_v = serde_json::to_value(raw)
            .map_err(|e| format!("failed to serialize verify result: {}", e))?;
        let summary = summarize_verify_like_output(&raw_v);

        let locs = plc::locate_sorries_in_file(&repo_root, &file, max_sorries, context_lines)?;
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
                json!({
                    "kind": "fix_first_error",
                    "prompt": rubberduck_prompt_first_error,
                    "line": first_error_line,
                })
            } else if nearest.is_some() {
                json!({
                    "kind": "patch_nearest_sorry",
                    "prompt": patch_prompt_nearest_sorry,
                    "region": nearest.as_ref().map(|s| json!({"start_line": s.region_start, "end_line": s.region_end})),
                })
            } else {
                json!({ "kind": "noop" })
            }
        };

        let full = json!({
            "repo_root": repo_root.display().to_string(),
            "file": file,
            "verify": { "summary": summary, "raw": raw_v },
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

    let mut out = lines.join("\n");
    if text.ends_with('\n') {
        out.push('\n');
    }
    (out, edits)
}

struct ProofyloopsAgentStepTool;

#[async_trait]
impl Tool for ProofyloopsAgentStepTool {
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
        let repo_root = repo_root_from_args(args)?;
        let file = extract_string(args, "file")?;
        let timeout_s = extract_u64_opt(args, "timeout_s")?.unwrap_or(180);
        let write = args.get("write").and_then(|v| v.as_bool()).unwrap_or(false);
        let output_path = extract_string_opt(args, "output_path");

        let repo_root = plc::find_lean_repo_root(&repo_root)?;
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

struct ProofyloopsReportHtmlTool;

#[async_trait]
impl Tool for ProofyloopsReportHtmlTool {
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

        let repo_root = plc::find_lean_repo_root(&repo_root)?;
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

            let ok = summary.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
            let counts = summary.get("counts").cloned().unwrap_or_else(|| json!({}));
            let errors = counts.get("errors").and_then(|v| v.as_u64()).unwrap_or(0);
            let warnings = counts.get("warnings").and_then(|v| v.as_u64()).unwrap_or(0);

            table.push(json!({
                "file": file,
                "verify": { "ok": ok, "errors": errors, "warnings": warnings },
                "sorries": { "count": locs.len(), "locations": locs },
            }));

            items.push(json!({
                "file": file,
                "verify": if include_raw_verify {
                    json!({ "summary": summary, "raw": raw_v })
                } else {
                    json!({ "summary": summary })
                },
                "sorries": { "count": locs.len(), "locations": locs },
            }));
        }

        let report_path = match output_path {
            Some(p) => std::path::PathBuf::from(p),
            None => std::env::temp_dir()
                .join(format!("proofloops-report-{}.html", uuid::Uuid::new_v4())),
        };
        if let Some(parent) = report_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create report dir {}: {}", parent.display(), e))?;
        }

        let mut html = String::new();
        html.push_str("<!doctype html>\n<html><head><meta charset=\"utf-8\"/>\n");
        html.push_str("<title>proofloops report</title>\n");
        html.push_str(
            "<style>body{font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial;max-width:1200px;margin:24px auto;padding:0 16px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px;vertical-align:top}th{background:#f6f6f6;text-align:left}code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,\"Liberation Mono\",monospace}pre{white-space:pre-wrap}</style>\n",
        );
        html.push_str("</head><body>\n");
        html.push_str("<h2>proofloops report</h2>\n");
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

struct ProofyloopsRubberduckPromptTool;

#[async_trait]
impl Tool for ProofyloopsRubberduckPromptTool {
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
                "proofloops_root": { "type": "string" },
                "proofyloops_root": { "type": "string", "description": "Legacy alias for proofloops_root" }
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

struct ProofyloopsLoopTool;

#[async_trait]
impl Tool for ProofyloopsLoopTool {
    fn description(&self) -> &str {
        "Bounded loop: suggest → patch first `sorry` in lemma → verify (`proofloops loop`)."
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
                "proofloops_root": { "type": "string" },
                "proofyloops_root": { "type": "string", "description": "Legacy alias for proofloops_root" }
            },
            "required": ["repo_root", "file", "lemma"]
        })
    }

    async fn call(&self, args: &Value) -> Result<Value, String> {
        let repo_root = repo_root_from_args(args)?;
        let file = extract_string(args, "file")?;
        let lemma = extract_string(args, "lemma")?;
        let max_iters = extract_u64_opt(args, "max_iters")?.unwrap_or(3);
        let timeout_s = extract_u64_opt(args, "timeout_s")?.unwrap_or(120);

        // Rust-native loop for suggest + patch + verify.
        // Keep `proofyloops_root` in the schema for backward compatibility, but ignore it.
        let _ = proofloops_root_from_args(args)?;
        let repo_root = plc::find_lean_repo_root(&repo_root)?;
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
#[derive(Clone)]
struct ProofyloopsStdioMcp {
    tool_router: RmcpToolRouter<Self>,
}

#[cfg(feature = "stdio")]
impl ProofyloopsStdioMcp {
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }
}

#[cfg(feature = "stdio")]
#[tool_router]
impl ProofyloopsStdioMcp {
    #[tool(description = "Triage a file: verify_summary + locate_sorries")]
    async fn proofyloops_triage_file(
        &self,
        params: Parameters<RepoFileSorriesArgs>,
    ) -> Result<CallToolResult, McpError> {
        let repo_root = PathBuf::from(&params.0.repo_root);
        let file = params.0.file.clone();
        let timeout_s = params.0.timeout_s.unwrap_or(180);
        let max_sorries = params.0.max_sorries.unwrap_or(50) as usize;
        let context_lines = params.0.context_lines.unwrap_or(1) as usize;
        let include_context_pack = params.0.include_context_pack.unwrap_or(true);
        let pack_context_lines = params.0.pack_context_lines.unwrap_or(6) as usize;
        let pack_nearby_lines = params.0.pack_nearby_lines.unwrap_or(60) as usize;
        let pack_max_nearby_decls = params.0.pack_max_nearby_decls.unwrap_or(20) as usize;
        let pack_max_imports = params.0.pack_max_imports.unwrap_or(20) as usize;
        let include_prompts = params.0.include_prompts.unwrap_or(true);
        let output_path = params.0.output_path.clone();

        let repo_root =
            plc::find_lean_repo_root(&repo_root).map_err(|e| McpError::invalid_params(e, None))?;
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
            "verify": { "summary": summary },
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

    #[tool(description = "Execute one safe agent step (no LLM): verify → mechanical fix → verify")]
    async fn proofyloops_agent_step(
        &self,
        params: Parameters<AgentStepArgs>,
    ) -> Result<CallToolResult, McpError> {
        let repo_root = PathBuf::from(&params.0.repo_root);
        let file = params.0.file.clone();
        let timeout_s = params.0.timeout_s.unwrap_or(180);
        let write = params.0.write.unwrap_or(false);
        let output_path = params.0.output_path.clone();

        let repo_root =
            plc::find_lean_repo_root(&repo_root).map_err(|e| McpError::invalid_params(e, None))?;
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
    async fn proofyloops_context_pack(
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

        let repo_root =
            plc::find_lean_repo_root(&repo_root).map_err(|e| McpError::invalid_params(e, None))?;
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
    async fn proofyloops_report_html(
        &self,
        params: Parameters<serde_json::Value>,
    ) -> Result<CallToolResult, McpError> {
        // Delegate to the existing Tool implementation to avoid duplicating report logic.
        let tool = ProofyloopsReportHtmlTool;
        let out = tool
            .call(&params.0)
            .await
            .map_err(|e| McpError::internal_error(e, None))?;
        Ok(CallToolResult::success(vec![Content::text(
            out.to_string(),
        )]))
    }
}

#[cfg(feature = "stdio")]
#[tool_handler]
impl rmcp::ServerHandler for ProofyloopsStdioMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "Tools for Lean proof triage/patching loops (proofyloops). JSON-only, stdout reserved for MCP frames."
                    .to_string(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Don’t emit logs to stdout if this ever becomes stdio-based.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Minimal transport switch:
    // - `proofyloops-mcp mcp-stdio` => stdio MCP server (Cursor can spawn it)
    // - default => HTTP MCP server (daemon)
    if std::env::args().nth(1).as_deref() == Some("mcp-stdio") {
        #[cfg(feature = "stdio")]
        {
            let service = ProofyloopsStdioMcp::new();
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

    let addr = std::env::var("PROOFLOOPS_MCP_ADDR")
        .or_else(|_| std::env::var("PROOFYLOOPS_MCP_ADDR"))
        .unwrap_or_else(|_| "127.0.0.1:8087".to_string());

    // The default axum-mcp tool timeout is 30s, but `lake build` / `lake env lean` can take longer
    // even on successful runs (big mathlib files + linters + first-time dependency builds).
    //
    // Override with: PROOFLOOPS_MCP_TOOL_TIMEOUT_S=900 (or larger)
    let tool_timeout_s: u64 = std::env::var("PROOFLOOPS_MCP_TOOL_TIMEOUT_S")
        .or_else(|_| std::env::var("PROOFYLOOPS_MCP_TOOL_TIMEOUT_S"))
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(180);
    let config = ServerConfig::new().with_tool_timeout(StdDuration::from_secs(tool_timeout_s));

    let server = McpServer::with_config(config)
        .tool("proofloops_prompt", ProofyloopsPromptTool)?
        .tool("proofloops_verify", ProofyloopsVerifyTool)?
        .tool("proofloops_verify_summary", ProofyloopsVerifySummaryTool)?
        .tool("proofloops_suggest", ProofyloopsSuggestTool)?
        .tool("proofloops_patch", ProofyloopsPatchTool)?
        .tool("proofloops_patch_region", ProofyloopsPatchRegionTool)?
        .tool("proofloops_locate_sorries", ProofyloopsLocateSorriesTool)?
        .tool("proofloops_context_pack", ProofyloopsContextPackTool)?
        .tool("proofloops_triage_file", ProofyloopsTriageFileTool)?
        .tool("proofloops_agent_step", ProofyloopsAgentStepTool)?
        .tool("proofloops_report_html", ProofyloopsReportHtmlTool)?
        .tool(
            "proofloops_rubberduck_prompt",
            ProofyloopsRubberduckPromptTool,
        )?
        .tool("proofloops_loop", ProofyloopsLoopTool)?;

    eprintln!("proofloops MCP server listening on http://{addr}");
    server.serve(&addr).await?;
    Ok(())
}
