use proofloops_core as plc;
use serde_json::json;
use std::path::PathBuf;
use std::time::Duration as StdDuration;
use std::{fs, io};

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

fn usage() -> String {
    [
        "proofloops (Rust) — direct CLI (no MCP).",
        "",
        "Commands:",
        "  triage-file --repo <path> --file <relpath> [--timeout-s N] [--max-sorries N] [--context-lines N] [--no-context-pack] [--no-prompts] [--output-json <path>]",
        "  agent-step  --repo <path> --file <relpath> [--timeout-s N] [--write] [--output-json <path>]",
        "  prompt      --repo <path> --file <relpath> --lemma <name> [--output-json <path>]",
        "  rubberduck-prompt --repo <path> --file <relpath> --lemma <name> [--diagnostics-file <path>] [--output-json <path>]",
        "  patch       --repo <path> --file <relpath> --lemma <name> --replacement-file <path> [--timeout-s N] [--output-json <path>]",
        "  suggest     --repo <path> --file <relpath> --lemma <name> [--timeout-s N] [--output-json <path>]",
        "  loop        --repo <path> --file <relpath> --lemma <name> [--max-iters N] [--timeout-s N] [--output-json <path>]",
        "  review-prompt --repo <path> [--scope staged|worktree] [--max-total-bytes N] [--per-file-bytes N] [--transcript-bytes N] [--cache-version STR] [--cache-model STR] [--output-json <path>]",
        "  review-diff --repo <path> [--scope staged|worktree] [--prompt-only] [--require-key] [--timeout-s N] [--max-total-bytes N] [--per-file-bytes N] [--transcript-bytes N] [--cache-version STR] [--cache-model STR] [--output-json <path>]",
        "  lint-style  --repo <path> [--github] --module <Root> [--module <Root> ...]",
        "  report      --repo <path> --files <relpath>... [--timeout-s N] [--max-sorries N] [--context-lines N] [--include-raw-verify] [--output-html <path>]",
        "  context-pack --repo <path> --file <relpath> [--decl <name> | --line N] [--context-lines N] [--nearby-lines N] [--max-nearby N] [--max-imports N]",
        "",
        "Notes:",
        "- Output is JSON to stdout.",
        "- This CLI uses proofloops-core, so verification runs `lake env lean` on the *real* file path.",
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

    match cmd {
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
            let stdout = raw_v.get("stdout").and_then(|v| v.as_str()).unwrap_or("");
            let stderr = raw_v.get("stderr").and_then(|v| v.as_str()).unwrap_or("");
            let first_error_loc = plc::parse_first_error_loc(stdout, stderr)
                .and_then(|loc| serde_json::to_value(loc).ok());

            let summary = {
                let ok = raw_v.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
                let timeout = raw_v
                    .get("timeout")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let returncode = raw_v
                    .get("returncode")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);

                let errors =
                    stdout.matches(": error:").count() + stderr.matches(": error:").count();
                let warnings =
                    stdout.matches(": warning:").count() + stderr.matches(": warning:").count();

                json!({
                    "ok": ok,
                    "timeout": timeout,
                    "returncode": returncode,
                    "counts": { "errors": errors, "warnings": warnings },
                    "first_error": stdout.lines().find(|l| l.contains(": error:")).or_else(|| stderr.lines().find(|l| l.contains(": error:"))),
                    "first_error_loc": first_error_loc
                })
            };

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
                            pack_max_nearby,
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

            let out = json!({
                "repo_root": repo_root.display().to_string(),
                "file": file,
                "verify": { "summary": summary, "raw": raw_v },
                "sorries": { "count": locs.len(), "locations": locs }
                ,
                "nearest_sorry_to_first_error": nearest,
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
            let verify = rt
                .block_on(plc::verify_lean_text(
                    &repo_root,
                    &patched.text,
                    StdDuration::from_secs(timeout_s),
                ))
                .map_err(|e| format!("verify failed: {e}"))?;

            let out = json!({
                "repo_root": repo_root.display().to_string(),
                "file": file,
                "lemma": lemma,
                "patch": {
                    "line": patched.line,
                    "before": patched.before,
                    "after": patched.after,
                    "indent": patched.indent,
                },
                "lemma_still_contains_sorry": still_has_sorry,
                "verify": verify,
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
            let scope = arg_value(rest, "--scope").unwrap_or_else(|| "staged".to_string());
            let max_total_bytes = arg_u64(rest, "--max-total-bytes").unwrap_or(180_000) as usize;
            let per_file_bytes = arg_u64(rest, "--per-file-bytes").unwrap_or(24_000) as usize;
            let transcript_bytes = arg_u64(rest, "--transcript-bytes").unwrap_or(24_000) as usize;
            let cache_version =
                arg_value(rest, "--cache-version").unwrap_or_else(|| "2026-01-19-v1".to_string());
            let cache_model =
                arg_value(rest, "--cache-model").unwrap_or_else(|| "unknown".to_string());
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);

            let scope_enum = match scope.as_str() {
                "staged" => plc::review::ReviewScope::Staged,
                "worktree" => plc::review::ReviewScope::Worktree,
                _ => return Err("scope must be staged|worktree".to_string()),
            };

            // Best-effort: keep progress off by default unless explicitly enabled.
            // (Review prompt building can run in pre-commit contexts.)
            let prompt = plc::review::build_review_prompt(
                &repo_root,
                scope_enum,
                max_total_bytes,
                per_file_bytes,
                transcript_bytes,
                &cache_model,
                &cache_version,
            )?;
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
            let scope = arg_value(rest, "--scope").unwrap_or_else(|| "worktree".to_string());
            let prompt_only = arg_flag(rest, "--prompt-only");
            let require_key = arg_flag(rest, "--require-key");
            let timeout_s = arg_u64(rest, "--timeout-s").unwrap_or(90);
            let max_total_bytes = arg_u64(rest, "--max-total-bytes").unwrap_or(180_000) as usize;
            let per_file_bytes = arg_u64(rest, "--per-file-bytes").unwrap_or(24_000) as usize;
            let transcript_bytes = arg_u64(rest, "--transcript-bytes").unwrap_or(24_000) as usize;
            let cache_version =
                arg_value(rest, "--cache-version").unwrap_or_else(|| "2026-01-19-v1".to_string());
            let cache_model =
                arg_value(rest, "--cache-model").unwrap_or_else(|| "unknown".to_string());
            let output_json = arg_value(rest, "--output-json").map(PathBuf::from);

            let scope_enum = match scope.as_str() {
                "staged" => plc::review::ReviewScope::Staged,
                "worktree" => plc::review::ReviewScope::Worktree,
                _ => return Err("scope must be staged|worktree".to_string()),
            };

            // Determine git root for env loading (LLM keys often live in <repo>/.env).
            let git_root = plc::review::git_repo_root(&repo_root)?;
            plc::load_dotenv_smart(&git_root);

            let prompt = plc::review::build_review_prompt(
                &git_root,
                scope_enum,
                max_total_bytes,
                per_file_bytes,
                transcript_bytes,
                &cache_model,
                &cache_version,
            )?;

            // Redact before emitting or sending.
            let diff = plc::review::redact_secrets(&prompt.diff);
            let corpus = plc::review::redact_secrets(&prompt.corpus);
            let transcript_tail = plc::review::redact_secrets(&prompt.transcript_tail);

            let system = [
                "You are a skeptical code reviewer for a Lean/mathlib-focused repository.",
                "Priorities:",
                "- correctness and proof soundness",
                "- API stability / portability (Linux case sensitivity, CI)",
                "- minimal diffs (prefer small fixes)",
                "Avoid praise. If unsure, say what is missing.",
                "Return a concise review with concrete, actionable fixes.",
                "",
            ]
            .join("\n");

            let payload = json!({
                "repo_root": prompt.repo_root,
                "scope": prompt.scope,
                "selected_files": prompt.selected_files,
                "diff": diff,
                "corpus": corpus,
                "transcript_tail": transcript_tail,
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

            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| format!("failed to build tokio runtime: {e}"))?;

            let res = rt.block_on(plc::llm::chat_completion(
                &system,
                &user,
                StdDuration::from_secs(timeout_s),
            ));

            let out = match res {
                Ok(r) => json!({
                    "repo_root": payload.get("repo_root").cloned().unwrap_or(serde_json::Value::Null),
                    "scope": payload.get("scope").cloned().unwrap_or(serde_json::Value::Null),
                    "provider": r.provider,
                    "model": r.model,
                    "review": r.content,
                    "cache_key": payload.get("cache_key").cloned().unwrap_or(serde_json::Value::Null),
                }),
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

                // Flatten into agent-ready next actions.
                for e in &error_samples {
                    next_actions.push(json!({
                        "kind": "fix_error",
                        "file": file,
                        "message": e,
                        "research": {
                            "mcp_calls": [
                                {
                                    "server": "user-arxiv-semantic-search-mcp",
                                    "tool": "search_papers",
                                    "arguments": { "query": format!("Lean 4 {}", e) }
                                }
                            ]
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
                            "mcp_calls": [
                                {
                                    "server": "user-arxiv-semantic-search-mcp",
                                    "tool": "search_papers",
                                    "arguments": { "query": q }
                                }
                            ]
                        }
                    }));
                }
                // Warnings last: they are useful, but proof-blocking work is usually sorries/errors.
                for w in &warning_samples {
                    next_actions.push(json!({
                        "kind": "fix_warning",
                        "file": file,
                        "message": w,
                        "research": {
                            "mcp_calls": [
                                {
                                    "server": "user-arxiv-semantic-search-mcp",
                                    "tool": "search_papers",
                                    "arguments": { "query": format!("Lean 4 {}", w) }
                                }
                            ]
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
                    "sorries": { "count": locs.len(), "locations": locs },
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
                html.push_str("<title>proofloops report</title>\n");
                html.push_str("<style>body{font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial;max-width:1200px;margin:24px auto;padding:0 16px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px;vertical-align:top}th{background:#f6f6f6;text-align:left}code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,\"Liberation Mono\",monospace}pre{white-space:pre-wrap}</style>\n");
                html.push_str("</head><body>\n");
                html.push_str("<h2>proofloops report</h2>\n");
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

        _ => Err(usage()),
    }
}
