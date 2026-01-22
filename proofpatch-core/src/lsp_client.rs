use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::str::FromStr;
use std::time::{Duration, Instant};

use lsp_types::{
    DidChangeTextDocumentParams,
    DidOpenTextDocumentParams,
    InitializeParams,
    LogMessageParams,
    PublishDiagnosticsParams,
    TextDocumentItem,
    Uri,
    VersionedTextDocumentIdentifier,
};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

// This is primarily a data carrier (returned to callers); fields may not be read inside this crate.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct LspDiag {
    pub uri: String,
    pub ok: bool,
    pub first_error_line_1: Option<usize>,
    pub first_error_col_1: Option<usize>,
    pub first_error_message: Option<String>,
    pub diagnostics_count: usize,
    /// Lean-ish diagnostic lines: `path:line:col: error|warning: msg`
    ///
    /// This exists to keep downstream parsing consistent with `lake env lean` output.
    pub lean_lines: Vec<String>,
    /// Best-effort log lines (e.g. from `window/logMessage`), used to recover `pp_dump` JSON
    /// and other Lean log output that doesn't appear in diagnostics.
    pub log_lines: Vec<String>,
    pub stderr: String,
    /// Time spent waiting for diagnostics (end-to-end) for this request.
    pub waited_ms: u64,
}

#[derive(Debug)]
struct PendingCheck {
    want_uri: String,
    diag: tokio::sync::oneshot::Sender<LspDiag>,
    deadline: Instant,
    last_diag: Option<PublishDiagnosticsParams>,
    last_update: Option<Instant>,
    started: Instant,
    log_lines: Vec<String>,
}

#[derive(Debug)]
struct ServerState {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    #[allow(dead_code)]
    next_id: u64,
    init_id: u64,
    ready_tx: Option<tokio::sync::oneshot::Sender<()>>,
    pending: Vec<PendingCheck>,
    /// Track opened documents by URI, with monotonically increasing version numbers.
    open_docs: HashMap<String, i32>,
    // Responses keyed by id; value is raw JSON for caller-specific parsing.
    resp_waiters: HashMap<u64, tokio::sync::oneshot::Sender<serde_json::Value>>,
}

static LSP_SERVERS: OnceLock<Mutex<HashMap<PathBuf, tokio::sync::mpsc::Sender<LspRequest>>>> =
    OnceLock::new();

static LSP_REPO_LOCKS: OnceLock<Mutex<HashMap<PathBuf, Arc<tokio::sync::Mutex<()>>>>> =
    OnceLock::new();

#[derive(Debug)]
enum LspRequest {
    CheckFile {
        #[allow(dead_code)]
        repo_root: PathBuf,
        file_path: PathBuf,
        text: String,
        timeout_s: Duration,
        resp: tokio::sync::oneshot::Sender<LspDiag>,
    },
}

fn lsp_uri_for_path(p: &Path) -> Result<Uri, String> {
    let u = url::Url::from_file_path(p).map_err(|_| format!("failed to build file:// uri for {}", p.display()))?;
    Uri::from_str(u.as_str()).map_err(|e| format!("failed to parse uri: {e}"))
}

fn encode_msg(v: &serde_json::Value) -> Vec<u8> {
    let body = v.to_string();
    let header = format!("Content-Length: {}\r\n\r\n", body.as_bytes().len());
    let mut out = header.into_bytes();
    out.extend_from_slice(body.as_bytes());
    out
}

async fn write_msg(stdin: &mut ChildStdin, v: &serde_json::Value) -> std::io::Result<()> {
    let bytes = encode_msg(v);
    stdin.write_all(&bytes).await?;
    stdin.flush().await?;
    Ok(())
}

async fn read_one_msg(stdout: &mut ChildStdout) -> std::io::Result<Option<serde_json::Value>> {
    // Read headers until blank line.
    let mut headers = Vec::<u8>::new();
    let mut buf = [0u8; 1];
    while !headers.ends_with(b"\r\n\r\n") {
        let n = stdout.read(&mut buf).await?;
        if n == 0 {
            return Ok(None);
        }
        headers.push(buf[0]);
        if headers.len() > 64 * 1024 {
            // header too large; treat as protocol error
            return Ok(None);
        }
    }
    let headers_s = String::from_utf8_lossy(&headers).to_string();
    let mut content_len: Option<usize> = None;
    for line in headers_s.split("\r\n") {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("Content-Length:") {
            content_len = rest.trim().parse::<usize>().ok();
        }
    }
    let Some(n) = content_len else {
        return Ok(None);
    };
    let mut body = vec![0u8; n];
    stdout.read_exact(&mut body).await?;
    let body_s = String::from_utf8_lossy(&body).to_string();
    match serde_json::from_str::<serde_json::Value>(&body_s) {
        Ok(v) => Ok(Some(v)),
        Err(_) => Ok(None),
    }
}

async fn pump_stdout(mut state: ServerState, mut rx: tokio::sync::mpsc::Receiver<LspRequest>) {
    // Start a small stderr pump (best-effort, bounded).
    let mut stderr = state.child.stderr.take();
    let stderr_buf: Arc<tokio::sync::Mutex<Vec<String>>> =
        Arc::new(tokio::sync::Mutex::new(Vec::<String>::new()));
    if let Some(mut s) = stderr.take() {
        let stderr_buf2 = Arc::clone(&stderr_buf);
        tokio::spawn(async move {
            let mut bytes = Vec::new();
            let _ = s.read_to_end(&mut bytes).await;
            let txt = String::from_utf8_lossy(&bytes).to_string();
            let mut g = stderr_buf2.lock().await;
            g.push(txt);
        });
    }

    // Heuristic: Lean LSP can publish diagnostics multiple times for the same file.
    // Wait a short "settle window" after the last diagnostics update before responding.
    const SETTLE_MS: u64 = 75;

    loop {
        tokio::select! {
            biased;

            Some(req) = rx.recv() => {
                match req {
                    LspRequest::CheckFile { repo_root: _, file_path, text, timeout_s, resp } => {
                        // Write text to disk first (Lean wants a real file path).
                        if let Some(parent) = file_path.parent() {
                            let _ = std::fs::create_dir_all(parent);
                        }
                        let _ = std::fs::write(&file_path, text.as_bytes());
                        let uri = match lsp_uri_for_path(&file_path) {
                            Ok(u) => u,
                            Err(e) => {
                                let _ = resp.send(LspDiag {
                                    uri: file_path.display().to_string(),
                                    ok: false,
                                    first_error_line_1: None,
                                    first_error_col_1: None,
                                    first_error_message: None,
                                    diagnostics_count: 0,
                                    lean_lines: Vec::new(),
                                log_lines: Vec::new(),
                                    stderr: e,
                                    waited_ms: 0,
                                });
                                continue;
                            }
                        };

                        let uri_s = uri.to_string();
                        if let Some(v) = state.open_docs.get_mut(&uri_s) {
                            // didChange (full text)
                            *v = v.saturating_add(1);
                            let params = DidChangeTextDocumentParams {
                                text_document: VersionedTextDocumentIdentifier {
                                    uri: uri.clone(),
                                    version: *v,
                                },
                                content_changes: vec![lsp_types::TextDocumentContentChangeEvent {
                                    range: None,
                                    range_length: None,
                                    text,
                                }],
                            };
                            let msg = json!({
                                "jsonrpc":"2.0",
                                "method": "textDocument/didChange",
                                "params": params,
                            });
                            let _ = write_msg(&mut state.stdin, &msg).await;
                        } else {
                            // didOpen
                            state.open_docs.insert(uri_s.clone(), 1);
                            let params = DidOpenTextDocumentParams {
                                text_document: TextDocumentItem {
                                    uri: uri.clone(),
                                    language_id: "lean".to_string(),
                                    version: 1,
                                    text,
                                },
                            };
                            let msg = json!({
                                "jsonrpc":"2.0",
                                "method": "textDocument/didOpen",
                                "params": params,
                            });
                            let _ = write_msg(&mut state.stdin, &msg).await;
                        }

                        // Register pending diagnostic waiter.
                            state.pending.push(PendingCheck {
                            want_uri: uri_s,
                            diag: resp,
                            deadline: Instant::now() + timeout_s,
                            last_diag: None,
                            last_update: None,
                                started: Instant::now(),
                                log_lines: Vec::new(),
                        });
                    }
                }
            }

            // Periodic sweep: finalize settled diagnostics and purge timed-out pendings.
            _ = tokio::time::sleep(Duration::from_millis(SETTLE_MS)) => {
                let now = Instant::now();
                let mut i = 0usize;
                while i < state.pending.len() {
                    let should_timeout = now >= state.pending[i].deadline;
                    let should_finalize = state.pending[i]
                        .last_update
                        .map(|t| now.duration_since(t) >= Duration::from_millis(SETTLE_MS))
                        .unwrap_or(false)
                        && state.pending[i].last_diag.is_some();

                    if should_timeout || should_finalize {
                        let mut pending = state.pending.remove(i);
                        let uri = pending.want_uri.clone();
                        let stderr_txt = {
                            let g = stderr_buf.lock().await;
                            g.join("\n")
                        };

                        let waited_ms = now.duration_since(pending.started).as_millis() as u64;
                        let (ok, first_line, first_col, first_msg, diag_count, lean_lines) =
                            if let Some(p) = pending.last_diag.take() {
                                let mut first_line = None;
                                let mut first_col = None;
                                let mut first_msg = None;
                                let mut err_count = 0usize;
                                let mut lean_lines: Vec<String> = Vec::new();
                                for d in &p.diagnostics {
                                    let sev = match d.severity {
                                        Some(lsp_types::DiagnosticSeverity::ERROR) => Some("error"),
                                        Some(lsp_types::DiagnosticSeverity::WARNING) => Some("warning"),
                                        _ => None,
                                    };
                                    let Some(sev_s) = sev else { continue; };
                                    let line_1 = d.range.start.line as usize + 1;
                                    let col_1 = d.range.start.character as usize + 1;
                                    let msg = d.message.replace('\n', " ");
                                    lean_lines.push(format!("{uri}:{line_1}:{col_1}: {sev_s}: {msg}"));
                                    if sev_s == "error" {
                                        err_count += 1;
                                        if first_line.is_none() {
                                            first_line = Some(line_1);
                                            first_col = Some(col_1);
                                            first_msg = Some(d.message.clone());
                                        }
                                    }
                                }
                                (
                                    err_count == 0,
                                    first_line,
                                    first_col,
                                    first_msg,
                                    p.diagnostics.len(),
                                    lean_lines,
                                )
                        } else {
                            // No diagnostics received yet; treat as timeout.
                            (false, None, None, None, 0usize, Vec::new())
                        };

                        let _ = pending.diag.send(LspDiag{
                            uri: uri.clone(),
                            ok,
                            first_error_line_1: first_line,
                            first_error_col_1: first_col,
                            first_error_message: first_msg,
                            diagnostics_count: diag_count,
                            lean_lines,
                            log_lines: pending.log_lines,
                            stderr: if should_timeout { format!("timeout waiting for diagnostics\n{stderr_txt}") } else { stderr_txt },
                            waited_ms,
                        });

                        // Intentionally keep documents open:
                        // - enables `didChange` reuse for a stable URI
                        // - avoids “didChange on a closed doc” bugs if our bookkeeping drifts
                        continue;
                    }

                    i += 1;
                }
            }

            msg = read_one_msg(&mut state.stdout) => {
                let Ok(msg) = msg else { break; };
                let Some(msg) = msg else { break; };

                // Handle responses
                if let Some(id) = msg.get("id").and_then(|v| v.as_u64()) {
                    if let Some(tx) = state.resp_waiters.remove(&id) {
                        let _ = tx.send(msg);
                        // After the initialize response, send `initialized` and mark ready.
                        if id == state.init_id {
                            let initd = json!({
                                "jsonrpc":"2.0",
                                "method": "initialized",
                                "params": {},
                            });
                            let _ = write_msg(&mut state.stdin, &initd).await;
                            if let Some(ready) = state.ready_tx.take() {
                                let _ = ready.send(());
                            }
                        }
                        continue;
                    }
                }

                // Handle publishDiagnostics
                if msg.get("method").and_then(|m| m.as_str()) == Some("textDocument/publishDiagnostics") {
                    if let Some(params) = msg.get("params") {
                        if let Ok(p) = serde_json::from_value::<PublishDiagnosticsParams>(params.clone()) {
                            // Update matching pending with latest diagnostics; finalization happens after settle.
                            if let Some(pend) = state.pending.iter_mut().find(|q| q.want_uri == p.uri.to_string()) {
                                pend.last_diag = Some(p);
                                pend.last_update = Some(Instant::now());
                            }
                        }
                    }
                }

                // Capture Lean log output. This is how we recover `pp_dump` JSON without spawning a process.
                if msg.get("method").and_then(|m| m.as_str()) == Some("window/logMessage") {
                    if let Some(params) = msg.get("params") {
                        if let Ok(p) = serde_json::from_value::<LogMessageParams>(params.clone()) {
                            // Attach logs to the most recent pending request (we serialize per repo, so this is safe).
                            if let Some(last) = state.pending.last_mut() {
                                // Keep it bounded.
                                if last.log_lines.len() < 2000 {
                                    last.log_lines.push(p.message);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

async fn start_server(repo_root: &Path, timeout_s: Duration) -> Result<tokio::sync::mpsc::Sender<LspRequest>, String> {
    // Spawn: lake env lean --server
    let lake = crate::resolve_lake();
    let mut cmd = Command::new(&lake);
    cmd.arg("env").arg("lean").arg("--server").current_dir(repo_root);
    crate::maybe_extend_lean_path_for_lake_env(&mut cmd);
    cmd.stdin(std::process::Stdio::piped());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| format!("failed to spawn lean --server: {e}"))?;
    let stdin = child.stdin.take().ok_or("missing stdin")?;
    let stdout = child.stdout.take().ok_or("missing stdout")?;

    let (tx, rx) = tokio::sync::mpsc::channel::<LspRequest>(16);
    // Initialize handshake: send initialize, then pump will send initialized and signal readiness.
    let init_id = 1u64;
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<()>();
    let mut state = ServerState {
        child,
        stdin,
        stdout,
        next_id: 2,
        init_id,
        ready_tx: Some(ready_tx),
        pending: Vec::new(),
        open_docs: HashMap::new(),
        resp_waiters: HashMap::new(),
    };

    let root_uri = lsp_uri_for_path(repo_root)?;
    let folder_name = repo_root
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("repo")
        .to_string();
    let params = InitializeParams {
        process_id: None,
        workspace_folders: Some(vec![lsp_types::WorkspaceFolder {
            uri: root_uri,
            name: folder_name,
        }]),
        capabilities: lsp_types::ClientCapabilities::default(),
        ..Default::default()
    };
    let init = json!({
        "jsonrpc":"2.0",
        "id": init_id,
        "method": "initialize",
        "params": params,
    });
    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    state.resp_waiters.insert(init_id, resp_tx);
    write_msg(&mut state.stdin, &init).await.map_err(|e| format!("failed to write initialize: {e}"))?;

    // Spawn pump loop
    tokio::spawn(pump_stdout(state, rx));

    // Wait for initialize response (bounded) and for pump to send initialized.
    let _ = tokio::time::timeout(timeout_s, resp_rx)
        .await
        .map_err(|_| "timeout waiting for initialize response".to_string())
        .and_then(|r| r.map_err(|_| "initialize response channel closed".to_string()))?;
    let _ = tokio::time::timeout(timeout_s, ready_rx)
        .await
        .map_err(|_| "timeout waiting for initialized to send".to_string())
        .and_then(|r| r.map_err(|_| "initialized readiness channel closed".to_string()))?;

    Ok(tx)
}

pub async fn check_text_via_lsp(
    repo_root: &Path,
    tmp_path: &Path,
    text: String,
    timeout_s: Duration,
) -> Result<LspDiag, String> {
    // Serialize checks per repo_root to avoid same-URI in-flight collisions.
    let locks = LSP_REPO_LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let lock = {
        let mut g = locks.lock().map_err(|_| "lsp locks cache lock poisoned".to_string())?;
        g.entry(repo_root.to_path_buf())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone()
    };
    // Bound lock acquisition too; otherwise a stuck prior request can cause unbounded waits.
    let _guard = tokio::time::timeout(timeout_s, lock.lock())
        .await
        .map_err(|_| "timeout waiting for lsp repo lock".to_string())?;

    let cache = LSP_SERVERS.get_or_init(|| Mutex::new(HashMap::new()));
    let key = repo_root.to_path_buf();
    let mut tx = {
        let mut g = cache.lock().map_err(|_| "lsp cache lock poisoned".to_string())?;
        if let Some(tx) = g.get(&key) {
            tx.clone()
        } else {
            let tx = start_server(repo_root, timeout_s).await?;
            g.insert(key.clone(), tx.clone());
            tx
        }
    };

    let text_retry = text.clone();
    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    let req = LspRequest::CheckFile {
        repo_root: repo_root.to_path_buf(),
        file_path: tmp_path.to_path_buf(),
        text,
        timeout_s,
        resp: resp_tx,
    };
    if tx.send(req).await.is_err() {
        // Server likely died; drop cache entry and retry once.
        {
            let mut g = cache.lock().map_err(|_| "lsp cache lock poisoned".to_string())?;
            g.remove(&key);
        }
        tx = start_server(repo_root, timeout_s).await?;
        {
            let mut g = cache.lock().map_err(|_| "lsp cache lock poisoned".to_string())?;
            g.insert(key.clone(), tx.clone());
        }

        let (resp2_tx, resp_rx2) = tokio::sync::oneshot::channel();
        let req2 = LspRequest::CheckFile {
            repo_root: repo_root.to_path_buf(),
            file_path: tmp_path.to_path_buf(),
            text: text_retry,
            timeout_s,
            resp: resp2_tx,
        };
        tx.send(req2)
            .await
            .map_err(|_| "lsp server task channel closed".to_string())?;
        let diag = tokio::time::timeout(timeout_s, resp_rx2)
            .await
            .map_err(|_| "timeout waiting for publishDiagnostics".to_string())?
            .map_err(|_| "lsp diag channel closed".to_string())?;
        return Ok(diag);
    }
    let diag = tokio::time::timeout(timeout_s, resp_rx)
        .await
        .map_err(|_| "timeout waiting for publishDiagnostics".to_string())?
        .map_err(|_| "lsp diag channel closed".to_string())?;
    Ok(diag)
}

