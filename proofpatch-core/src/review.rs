use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime};

/// A small, structured progress event (JSONL-friendly).
///
/// Public invariants:
/// - `phase` is a small enum-like string, stable-ish for downstream parsing.
/// - `ts` is ISO-ish UTC (best-effort).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressEvent {
    pub event: String,
    pub ts: String,
    pub phase: String,
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub data: serde_json::Value,
}

fn now_iso() -> String {
    // Keep it dependency-free. This is not strict RFC3339; it's “good enough”.
    // Format: seconds since epoch, plus a "Z" marker.
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0))
        .as_secs();
    format!("{secs}Z")
}

fn env_truthy(name: &str, default_on: bool) -> bool {
    let v = std::env::var(name).ok().unwrap_or_default();
    let v = v.trim().to_lowercase();
    if v.is_empty() {
        return default_on;
    }
    !matches!(v.as_str(), "0" | "false" | "no" | "off")
}

/// Emit progress events to stderr.
///
/// Controls:
/// - `PROOFPATCH_REVIEW_PROGRESS`:
///   - "pretty" (default) prints one-line human logs
///   - "jsonl" prints JSONL lines
///   - "pretty,jsonl" does both
///   - "0/off/false/no" disables
pub fn emit_progress(root: &Path, mut ev: ProgressEvent) {
    let mode = std::env::var("PROOFPATCH_REVIEW_PROGRESS")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "pretty".to_string());
    if matches!(mode.as_str(), "0" | "off" | "false" | "no") {
        return;
    }
    if ev.ts.trim().is_empty() {
        ev.ts = now_iso();
    }

    let want_pretty = mode.split(',').any(|s| s.trim() == "pretty");
    let want_jsonl = mode.split(',').any(|s| s.trim() == "jsonl");

    if want_pretty {
        let msg = ev.message.trim();
        if msg.is_empty() {
            eprintln!("review[{}]", ev.phase);
        } else {
            eprintln!("review[{}]: {}", ev.phase, msg);
        }
    }
    if want_jsonl {
        if let Ok(line) = serde_json::to_string(&ev) {
            eprintln!("{line}");
        }
        // Optional JSONL file output (git-ignored location).
        let p = std::env::var("PROOFPATCH_REVIEW_PROGRESS_FILE")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join(".git").join("proofpatch_review_progress.jsonl"));
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(line) = serde_json::to_string(&ev) {
            let _ = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&p)
                .and_then(|mut f| {
                    std::io::Write::write_all(&mut f, format!("{line}\n").as_bytes())
                });
        }
    }
}

/// Best-effort redaction of obvious secret patterns.
pub fn redact_secrets(s: &str) -> String {
    if s.trim().is_empty() {
        return s.to_string();
    }
    let mut out = s.to_string();
    // KEY=VALUE patterns
    let re_kv =
        regex::Regex::new(r"(OPENROUTER_API_KEY|OPENAI_API_KEY|GROQ_API_KEY)\s*=\s*[^\s]+").ok();
    if let Some(re) = re_kv {
        out = re.replace_all(&out, "$1=[REDACTED]").to_string();
    }
    // Authorization: Bearer <token>
    let re_auth = regex::Regex::new(r"(?i)(Authorization:\s*Bearer)\s+[^\s]+").ok();
    if let Some(re) = re_auth {
        out = re.replace_all(&out, "$1 [REDACTED]").to_string();
    }
    // OpenAI-ish tokens
    let re_sk = regex::Regex::new(r"\b(sk-[A-Za-z0-9_\-]{16,})\b").ok();
    if let Some(re) = re_sk {
        out = re.replace_all(&out, "[REDACTED_TOKEN]").to_string();
    }
    // GitHub tokens / PATs (common accidental leaks)
    let re_gh = regex::Regex::new(r"\b(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b").ok();
    if let Some(re) = re_gh {
        out = re.replace_all(&out, "[REDACTED_TOKEN]").to_string();
    }
    // AWS access key id (very coarse; avoids obvious leaks)
    let re_aws = regex::Regex::new(r"\b(AKIA[0-9A-Z]{16})\b").ok();
    if let Some(re) = re_aws {
        out = re.replace_all(&out, "[REDACTED_TOKEN]").to_string();
    }
    out
}

/// Include a small tail of the most recent Cursor agent transcript, if present.
///
/// Controls:
/// - `PROOFPATCH_REVIEW_INCLUDE_TRANSCRIPT` (default: off)
/// - `PROOFPATCH_REVIEW_TRANSCRIPT_PATH` (explicit file path override)
pub fn agent_transcript_tail(max_bytes: usize) -> String {
    // Default OFF: transcript tails often contain unrelated tool output and may include secrets.
    if !env_truthy("PROOFPATCH_REVIEW_INCLUDE_TRANSCRIPT", false) {
        return String::new();
    }
    let explicit = std::env::var("PROOFPATCH_REVIEW_TRANSCRIPT_PATH")
        .ok()
        .unwrap_or_default()
        .trim()
        .to_string();

    let mut candidates: Vec<PathBuf> = Vec::new();
    if !explicit.is_empty() {
        let p = PathBuf::from(explicit);
        if p.is_file() {
            candidates.push(p);
        }
    } else {
        // Cheap, shallow scan:
        // ~/.cursor/projects/<workspace>/agent-transcripts/*.txt
        if let Some(home) = dirs::home_dir() {
            let base = home.join(".cursor").join("projects");
            if base.is_dir() {
                if let Ok(entries) = std::fs::read_dir(&base) {
                    for e in entries.flatten() {
                        let ws = e.path();
                        if !ws.is_dir() {
                            continue;
                        }
                        let at = ws.join("agent-transcripts");
                        if !at.is_dir() {
                            continue;
                        }
                        if let Ok(files) = std::fs::read_dir(&at) {
                            for f in files.flatten() {
                                let p = f.path();
                                if p.extension().and_then(|s| s.to_str()) == Some("txt")
                                    && p.is_file()
                                {
                                    candidates.push(p);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if candidates.is_empty() {
        return String::new();
    }
    candidates.sort_by_key(|p| {
        p.metadata()
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH)
    });
    let p = candidates.last().cloned().unwrap();

    let raw = match std::fs::read(&p) {
        Ok(b) => b,
        Err(_) => return String::new(),
    };
    let tail = if raw.len() <= max_bytes {
        raw.as_slice()
    } else {
        &raw[raw.len() - max_bytes..]
    };
    let mut txt = String::from_utf8_lossy(tail).to_string();
    txt = redact_secrets(&txt);
    // Ensure the final inclusion is bounded even after formatting.
    if txt.as_bytes().len() > max_bytes {
        txt = String::from_utf8_lossy(&txt.as_bytes()[..max_bytes]).to_string();
    }
    format!(
        "===== AGENT_TRANSCRIPT_TAIL {} (bytes={}, tail_bytes={}) =====\n{}\n",
        p.file_name().and_then(|s| s.to_str()).unwrap_or("unknown"),
        raw.len(),
        tail.len(),
        txt
    )
}

/// Conservative “never send this” filter.
pub fn is_sensitive_path(root: &Path, p: &Path) -> bool {
    let rp = p;
    let rel_parts: Vec<String> = rp
        .strip_prefix(root)
        .unwrap_or(rp)
        .components()
        .map(|c| c.as_os_str().to_string_lossy().to_lowercase())
        .collect();

    if rel_parts
        .iter()
        .any(|x| x == ".git" || x == ".lake" || x == "lake-packages")
    {
        return true;
    }
    let name = rp
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    if name == ".env" || name.starts_with(".env.") || name == ".envrc" {
        return true;
    }
    if name.contains("id_rsa") || name.contains("id_ed25519") {
        return true;
    }
    if name.ends_with(".pem")
        || name.ends_with(".key")
        || name.ends_with(".p12")
        || name.ends_with(".pfx")
        || name.ends_with(".gpg")
        || name.ends_with(".asc")
    {
        return true;
    }
    let ext = rp
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    if matches!(
        ext.as_str(),
        "pdf" | "png" | "jpg" | "jpeg" | "gif" | "webp" | "zip" | "gz" | "xz" | "bz2"
    ) {
        return true;
    }
    false
}

pub fn filter_review_paths(root: &Path, paths: Vec<PathBuf>) -> Vec<PathBuf> {
    paths
        .into_iter()
        .filter(|p| !is_sensitive_path(root, p))
        .collect()
}

fn is_noisy_excerpt_path(rel: &str) -> bool {
    // Keep diffs, but don't embed huge lockfiles/config dumps as “context excerpts”.
    let base = rel
        .replace('\\', "/")
        .split('/')
        .last()
        .unwrap_or("")
        .to_lowercase();
    matches!(
        base.as_str(),
        "uv.lock"
            | "cargo.lock"
            | "package-lock.json"
            | "pnpm-lock.yaml"
            | "yarn.lock"
            | "poetry.lock"
    )
}

// Intentionally no generic shell helper here; keep this module deterministic and low-risk.

pub fn git_repo_root(start: &Path) -> Result<PathBuf, String> {
    let out = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(start)
        .output()
        .map_err(|e| format!("failed to run git: {e}"))?;
    if !out.status.success() {
        return Err("git rev-parse --show-toplevel failed".to_string());
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        return Err("empty git repo root".to_string());
    }
    Ok(PathBuf::from(s))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileBlob {
    pub rel: String,
    pub bytes_len: usize,
    pub sha256: String,
    pub content: String,
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlobMeta {
    pub rel: String,
    pub bytes_len: usize,
    pub sha256: String,
    pub truncated: bool,
}

pub fn read_blob(root: &Path, p: &Path, max_bytes: usize) -> Result<FileBlob, String> {
    let raw = std::fs::read(p).map_err(|e| format!("read {}: {e}", p.display()))?;
    let bytes_len = raw.len();
    let truncated = raw.len() > max_bytes;
    let marker = b"\n\n[...TRUNCATED: middle omitted...]\n\n";
    let raw2: Vec<u8> = if raw.len() > max_bytes && max_bytes > marker.len() + 8 {
        // For large files, a prefix-only excerpt is actively misleading: it can cut a declaration
        // header or omit the tail where the “real work” is. Prefer prefix + suffix excerpts.
        //
        // This makes LLM review outputs less likely to hallucinate “file is truncated mid-lemma”.
        let keep = max_bytes - marker.len();
        let head = keep / 2;
        let tail = keep - head;
        let mut v = Vec::with_capacity(max_bytes);
        v.extend_from_slice(&raw[..head]);
        v.extend_from_slice(marker);
        v.extend_from_slice(&raw[raw.len() - tail..]);
        v
    } else if raw.len() > max_bytes {
        raw[..max_bytes].to_vec()
    } else {
        raw
    };
    let mut h = Sha256::new();
    h.update(&raw2);
    let mut digest = hex::encode(h.finalize());
    if truncated {
        digest = format!("excerpt:{digest}");
    }
    let content = redact_secrets(&String::from_utf8_lossy(&raw2).to_string());
    let rel = p
        .strip_prefix(root)
        .unwrap_or(p)
        .to_string_lossy()
        .to_string();
    Ok(FileBlob {
        rel,
        bytes_len,
        sha256: digest,
        content,
        truncated,
    })
}

pub fn assemble_corpus(blobs: &[FileBlob], max_total_bytes: usize) -> (String, usize) {
    let mut total = 0usize;
    let mut chunks: Vec<String> = Vec::new();
    for b in blobs {
        let header = format!(
            "===== {} (bytes={}, sha256={}{} ) =====\n",
            b.rel,
            b.bytes_len,
            b.sha256,
            if b.truncated { ", TRUNCATED" } else { "" }
        );
        let mut piece = header;
        piece.push_str(&b.content);
        if !b.content.ends_with('\n') {
            piece.push('\n');
        }
        piece.push('\n');
        let piece_bytes = piece.as_bytes().len();
        if total + piece_bytes > max_total_bytes {
            break;
        }
        chunks.push(piece);
        total += piece_bytes;
    }
    (chunks.concat(), total)
}

pub fn cache_key(
    version: &str,
    model: &str,
    scope: &str,
    diff_txt: &str,
    blobs: &[FileBlob],
) -> String {
    let mut h = Sha256::new();
    h.update(version.as_bytes());
    h.update(b"\0");
    h.update(model.as_bytes());
    h.update(b"\0");
    h.update(scope.as_bytes());
    h.update(b"\0");
    h.update(diff_txt.as_bytes());
    h.update(b"\0");
    for b in blobs {
        h.update(b.rel.as_bytes());
        h.update(b"\0");
        h.update(b.sha256.as_bytes());
        h.update(b"\0");
    }
    hex::encode(h.finalize())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewPrompt {
    pub repo_root: String,
    pub scope: String,
    pub max_total_bytes: usize,
    pub blob_meta: Vec<BlobMeta>,
    pub diff: String,
    pub corpus: String,
    pub transcript_tail: String,
    pub selected_files: Vec<String>,
    pub cache_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReviewScope {
    Staged,
    Worktree,
}

fn git_lines(root: &Path, args: &[&str]) -> Result<Vec<String>, String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|e| format!("git {:?}: {e}", args))?;
    if !out.status.success() {
        return Ok(vec![]);
    }
    let s = String::from_utf8_lossy(&out.stdout).to_string();
    Ok(s.lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect())
}

fn git_text(root: &Path, args: &[&str]) -> Result<String, String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|e| format!("git {:?}: {e}", args))?;
    if !out.status.success() {
        return Ok(String::new());
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

pub fn build_review_prompt(
    repo_root: &Path,
    scope: ReviewScope,
    max_total_bytes: usize,
    per_file_max_bytes: usize,
    transcript_max_bytes: usize,
    model_for_cache: &str,
    cache_version: &str,
) -> Result<ReviewPrompt, String> {
    let repo_root = git_repo_root(repo_root)?;

    let (paths, diff_txt, scope_name) = match scope {
        ReviewScope::Staged => {
            let files = git_lines(
                &repo_root,
                &["diff", "--cached", "--name-only", "--diff-filter=ACMRT"],
            )?;
            let diff = git_text(&repo_root, &["diff", "--cached"])?;
            (files, diff, "staged".to_string())
        }
        ReviewScope::Worktree => {
            let staged = git_lines(
                &repo_root,
                &["diff", "--cached", "--name-only", "--diff-filter=ACMRT"],
            )?;
            let unstaged = git_lines(&repo_root, &["diff", "--name-only", "--diff-filter=ACMRT"])?;
            let untracked = git_lines(&repo_root, &["ls-files", "--others", "--exclude-standard"])?;
            let mut set: BTreeSet<String> = BTreeSet::new();
            for x in staged.into_iter().chain(unstaged).chain(untracked) {
                set.insert(x);
            }
            let files = set.into_iter().collect::<Vec<_>>();
            let diff = {
                let a = git_text(&repo_root, &["diff", "--cached"])?;
                let b = git_text(&repo_root, &["diff"])?;
                if !a.is_empty() && !b.is_empty() {
                    format!("{a}\n\n{b}")
                } else {
                    format!("{a}{b}")
                }
            };
            (files, diff, "worktree".to_string())
        }
    };
    let diff_txt = redact_secrets(&diff_txt);

    // Enforce a *total* prompt budget across:
    // - the raw git diff text,
    // - the selected file corpus excerpts,
    // - and an optional agent transcript tail.
    //
    // Historically, `max_total_bytes` only bounded the corpus, which meant the diff could explode
    // and overwhelm provider context windows. Treat `max_total_bytes` as the overall cap.
    fn truncate_to_bytes_lossy(s: &str, max_bytes: usize) -> String {
        if max_bytes == 0 {
            return String::new();
        }
        let b = s.as_bytes();
        if b.len() <= max_bytes {
            return s.to_string();
        }
        // Keep it UTF-8 safe (best-effort).
        let mut out = String::from_utf8_lossy(&b[..max_bytes]).to_string();
        out.push_str("\n…[TRUNCATED]\n");
        out
    }

    // Filter + read blobs (bounded).
    let mut kept: Vec<PathBuf> = Vec::new();
    for rel in &paths {
        let p = repo_root.join(rel);
        if p.is_file() && !is_sensitive_path(&repo_root, &p) && !is_noisy_excerpt_path(rel) {
            kept.push(p);
        }
    }
    let selected_files: Vec<String> = kept
        .iter()
        .map(|p| {
            p.strip_prefix(&repo_root)
                .unwrap_or(p)
                .to_string_lossy()
                .to_string()
        })
        .collect();

    let mut blobs: Vec<FileBlob> = Vec::new();
    for p in &kept {
        if let Ok(b) = read_blob(&repo_root, p, per_file_max_bytes) {
            blobs.push(b);
        }
    }
    let blob_meta = blobs
        .iter()
        .map(|b| BlobMeta {
            rel: b.rel.clone(),
            bytes_len: b.bytes_len,
            sha256: b.sha256.clone(),
            truncated: b.truncated,
        })
        .collect::<Vec<_>>();
    // Budget split (stable + conservative):
    // - ~40% diff
    // - ~50% corpus
    // - ~10% transcript tail (if enabled)
    //
    // This keeps "what changed" (diff) and "local context" (corpus) both present by default.
    let total_budget = max_total_bytes.max(1);
    let transcript_budget = usize::min(transcript_max_bytes, usize::max(1, total_budget / 10));
    let diff_budget = usize::max(1, (total_budget * 2) / 5);
    let mut corpus_budget = total_budget.saturating_sub(diff_budget + transcript_budget);
    if corpus_budget == 0 {
        corpus_budget = 1;
    }

    let diff_txt = truncate_to_bytes_lossy(&diff_txt, diff_budget);
    let (corpus, _bytes) = assemble_corpus(&blobs, corpus_budget);
    let transcript_tail = agent_transcript_tail(transcript_budget);
    let ck = cache_key(
        cache_version,
        model_for_cache,
        &scope_name,
        &diff_txt,
        &blobs,
    );

    Ok(ReviewPrompt {
        repo_root: repo_root.display().to_string(),
        scope: scope_name,
        max_total_bytes,
        blob_meta,
        diff: diff_txt,
        corpus,
        transcript_tail,
        selected_files,
        cache_key: ck,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redact_secrets_redacts_common_patterns() {
        let s = "OPENROUTER_API_KEY=abc\nGROQ_API_KEY=def\nAuthorization: Bearer sk-1234567890abcdef\nsk-1234567890abcdef";
        let r = redact_secrets(s);
        assert!(!r.contains("abc"));
        assert!(!r.contains("def"));
        assert!(r.contains("OPENROUTER_API_KEY=[REDACTED]"));
        assert!(r.contains("GROQ_API_KEY=[REDACTED]"));
        assert!(r
            .to_lowercase()
            .contains("authorization: bearer [redacted]"));
        assert!(r.contains("[REDACTED_TOKEN]"));
    }

    #[test]
    fn cache_key_changes_when_blob_changes() {
        let b1 = FileBlob {
            rel: "a.txt".into(),
            bytes_len: 3,
            sha256: "x".into(),
            content: "aaa".into(),
            truncated: false,
        };
        let b2 = FileBlob {
            sha256: "y".into(),
            ..b1.clone()
        };
        let k1 = cache_key("v1", "m", "staged", "diff", &[b1]);
        let k2 = cache_key("v1", "m", "staged", "diff", &[b2]);
        assert_ne!(k1, k2);
    }
}
