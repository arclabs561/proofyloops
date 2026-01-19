# proofyloops

Local Lean “proof helper” that can grow into a prover.

This is not a full automated prover yet. The initial scope is:

- a small **OpenAI-compatible client** that can talk to:
  - Ollama (local)
  - Groq
  - OpenRouter
- a **proof-suggestion** command that takes a Lean lemma statement and proposes a `by ...` proof block

## Provider configuration (opportunistic)

The client will try providers in `PROOFYLOOPS_PROVIDER_ORDER` (default: `ollama,groq,openai,openrouter`),
skipping providers whose required env vars are not set / not reachable.

### Ollama

- `OLLAMA_HOST` (optional, default `http://localhost:11434`)
- `OLLAMA_MODEL` (required)

### Groq

- `GROQ_API_KEY` (required)
- `GROQ_MODEL` (required)

### OpenAI (or OpenAI-compatible)

- `OPENAI_API_KEY` (required)
- `OPENAI_MODEL` (required)
- optional: `OPENAI_BASE_URL` (default: `https://api.openai.com/v1`)

### OpenRouter

- `OPENROUTER_API_KEY` (required)
- `OPENROUTER_MODEL` (required)
- optional: `OPENROUTER_SITE_URL`, `OPENROUTER_APP_NAME`

### Environment loading (super-workspace convenience)

`proofyloops-core` loads environment variables in a “never override” order:

- `<repo_root>/.env` (if present)
- `~/.cursor/mcp.json` `mcpServers.*.env` blocks (if present)
- optional sibling `.env` search (one directory deep), if no key is set yet

Controls:
- `PROOFYLOOPS_DOTENV_SEARCH=0` disables the sibling search (default: enabled)
- `PROOFYLOOPS_DOTENV_SEARCH_ROOT=/abs/path` overrides the sibling-search root (default: `repo_root.parent`)
- `PROOFYLOOPS_MCP_JSON_PATH=/abs/path/to/mcp.json` override (primarily for tests)
- `PROOFYLOOPS_AUTO_BUILD=0` disables auto `lake build` on missing `.olean` (default: enabled)

## Usage

Show help:

```bash
uvx --from /Users/arc/Documents/dev/proofyloops proofyloops --help
```

### Rust direct CLI (no Python, no MCP)

For agent loops that want **JSON-only** and don’t want to rely on `uvx`/Python, there is a small
Rust CLI binary in `proofyloops-core`:

```bash
cd /Users/arc/Documents/dev/proofyloops/proofyloops-core
cargo run --quiet --bin proofyloops -- --help
```

Triage a file (verify summary + `sorry` locations):

```bash
cd /Users/arc/Documents/dev/proofyloops/proofyloops-core
cargo run --quiet --bin proofyloops -- triage-file \
  --repo /Users/arc/Documents/dev/geometry-of-numbers \
  --file Covolume/Cauchy/Main.lean
```

By default `triage-file` also includes:
- a **rubberduck prompt payload** targeted at the first error line (if any)
- a **region patch prompt payload** targeted at the nearest `sorry` region (if any)
- `next_action`: a single recommended next move (fix the first error, or patch the nearest sorry)

Disable those (smaller JSON):

```bash
cd /Users/arc/Documents/dev/proofyloops/proofyloops-core
cargo run --quiet --bin proofyloops -- triage-file \
  --repo /Users/arc/Documents/dev/geometry-of-numbers \
  --file Covolume/Cauchy/Main.lean \
  --no-prompts
```

Write the full JSON to a file (stdout becomes a small summary with `written` path):

```bash
cd /Users/arc/Documents/dev/proofyloops/proofyloops-core
cargo run --quiet --bin proofyloops -- triage-file \
  --repo /Users/arc/Documents/dev/geometry-of-numbers \
  --file Covolume/Cauchy/Main.lean \
  --output-json /tmp/proofyloops-triage.json
```

Build a “context pack” (imports + focused excerpt + nearby decl headers):

```bash
cd /Users/arc/Documents/dev/proofyloops/proofyloops-core
cargo run --quiet --bin proofyloops -- context-pack \
  --repo /Users/arc/Documents/dev/geometry-of-numbers \
  --file Covolume/Cauchy/Main.lean \
  --decl cauchy_decomposition
```

Execute one **safe agent step** (no LLM; intended to unblock trivial failures):

- Runs `lake env lean` (verify)
- If the first error message suggests a mechanical fix (currently: `ring` → `ring_nf`), applies it
- Re-verifies

Dry-run (does **not** write the file; verifies via a temp file):

```bash
cd /Users/arc/Documents/dev/proofyloops/proofyloops-core
cargo run --quiet --bin proofyloops -- agent-step \
  --repo /Users/arc/Documents/dev/geometry-of-numbers \
  --file Covolume/Cauchy/Main.lean \
  --output-json /tmp/proofyloops-agent-step.json
```

Write back to the file:

```bash
cd /Users/arc/Documents/dev/proofyloops/proofyloops-core
cargo run --quiet --bin proofyloops -- agent-step \
  --repo /Users/arc/Documents/dev/geometry-of-numbers \
  --file Covolume/Cauchy/Main.lean \
  --write \
  --output-json /tmp/proofyloops-agent-step-write.json
```

### MCP tool: `proofyloops_agent_step`

Available in both HTTP and stdio servers. It mirrors the CLI behavior: verify → mechanical fix → verify,
and returns a small DAG trace.

Arguments:
- `repo_root` (string, required)
- `file` (string, required)
- `timeout_s` (int, optional)
- `write` (bool, optional; default false)
- `output_path` (string, optional; if set, writes full JSON and returns a small summary)

Print the prompt payload (no LLM call):

```bash
uvx --from /Users/arc/Documents/dev/proofyloops proofyloops prompt \
  --repo /Users/arc/Documents/dev/geometry-of-numbers \
  --file Covolume/Legendre/Ankeny.lean \
  --lemma ankeny_even_padicValNat_of_mem_primeFactors
```

Verify a Lean file elaborates (via `lake env lean`):

```bash
uvx --from /Users/arc/Documents/dev/proofyloops proofyloops verify \
  --repo /Users/arc/Documents/dev/geometry-of-numbers \
  --file Covolume/Legendre/Ankeny.lean
```

Suggest a proof for a lemma in a file:

```bash
uvx --from /Users/arc/Documents/dev/proofyloops proofyloops suggest \
  --repo /Users/arc/Documents/dev/geometry-of-numbers \
  --file Covolume/Legendre/Ankeny.lean \
  --lemma reduction_to_sum_three_squares
```

Apply a replacement (from a file) and verify (no LLM call):

```bash
uvx --from /Users/arc/Documents/dev/proofyloops proofyloops patch \
  --repo /Users/arc/Documents/dev/geometry-of-numbers \
  --file Covolume/Legendre/Ankeny.lean \
  --lemma ankeny_even_padicValNat_of_mem_primeFactors \
  --replacement-file /tmp/replacement.lean
```

Bounded loop (suggest → patch first `sorry` in lemma → verify):

```bash
uvx --from /Users/arc/Documents/dev/proofyloops proofyloops loop \
  --repo /Users/arc/Documents/dev/geometry-of-numbers \
  --file Covolume/Legendre/Ankeny.lean \
  --lemma reduction_to_sum_three_squares \
  --max-iters 3
```

Review the repo diff with an LLM (bounded; skips sensitive paths; exits 0 if unconfigured unless `--require-key`):

```bash
uvx --from /Users/arc/Documents/dev/proofyloops proofyloops review-diff \
  --repo /Users/arc/Documents/dev/geometry-of-numbers \
  --scope unstaged
```

Build a bounded review context pack for a git diff (no LLM call; safe-path filtering + optional transcript tail):

```bash
cd /Users/arc/Documents/dev/proofyloops/proofyloops-core
cargo run --quiet --bin proofyloops -- review-prompt \
  --repo /Users/arc/Documents/dev/geometry-of-numbers \
  --scope worktree \
  --max-total-bytes 180000 \
  --per-file-bytes 24000 \
  --output-json /tmp/proofyloops-review-prompt.json
```

## MCP server (axum-mcp)

There is a small Rust HTTP server that exposes `proofyloops` as MCP tools, implemented using the
top-level `axum-mcp` crate in this dev workspace.

Most “file surgery + verification” logic lives in `proofyloops-core` (Rust). The MCP server is now
Rust-native for provider routing too (it does not shell out to `uvx` for `suggest`/`loop`).

Run:

```bash
cd /Users/arc/Documents/dev/proofyloops/mcp-server
cargo run
```

Controls:
- `PROOFYLOOPS_MCP_ADDR` (default: `127.0.0.1:8087`)
- `PROOFYLOOPS_MCP_TOOL_TIMEOUT_S` (default: `180`) — for first-time `lake build` in fresh repos, you may want `900`+

Then:

```bash
curl http://127.0.0.1:8087/health
curl http://127.0.0.1:8087/tools/list
curl -X POST http://127.0.0.1:8087/tools/call -H 'Content-Type: application/json' \
  -d '{"name":"proofyloops_prompt","arguments":{"repo_root":"/Users/arc/Documents/dev/geometry-of-numbers","file":"Covolume/Legendre/Ankeny.lean","lemma":"ankeny_even_padicValNat_of_mem_primeFactors"}}'
```

Tools:
- `proofyloops_prompt`: extract lemma excerpt + prompt payload (no LLM call)
- `proofyloops_suggest`: call the configured provider to get Lean proof text
- `proofyloops_patch`: splice replacement into the lemma’s first `sorry` and verify (does not write to disk)
- `proofyloops_patch_region`: splice replacement into the first `sorry` within a line region (does not write to disk)
- `proofyloops_locate_sorries`: locate `sorry` tokens in a file with suggested patch regions
- `proofyloops_triage_file`: `verify_summary` + `locate_sorries` in one call (plus nearest sorry to first error if any)
- `proofyloops_verify`: verify a file elaborates (does not write to disk)
- `proofyloops_verify_summary`: like `verify`, but returns `{summary, raw}` where summary includes counts + first error line
- `proofyloops_rubberduck_prompt`: build a “planning-only” prompt (no Lean code) to decide the next small proof step
- `proofyloops_loop`: bounded loop (suggest → patch → verify), returning attempts + last verify output

Notes:
- **Lean 4 only (for now)**: if a repo has `leanpkg.toml`, we treat it as Lean 3 and error early.
- **Patch semantics**: we replace the first `sorry` *token* on the matched line (not the whole line).
  - Example (`proofyloops_patch_region`):
    - before: `def double (n : Nat) : Nat := sorry`
    - after:  `def double (n : Nat) : Nat := n + n`
  - Example (`proofyloops_patch_region` on an instance field):
    - before: `pure_bind := by sorry`
    - after:  `pure_bind := by simp`
  - If the original line already contains `... := by sorry` and your replacement starts with `by ...`,
    we strip the leading `by` to avoid generating `by by ...`.

Cursor integration:
- add an MCP server entry in `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "proofyloops": { "url": "http://127.0.0.1:8087" }
  }
}
```

