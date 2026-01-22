## proofpatch

Rust helpers for working in Lean 4 repos: verify, locate `sorry`s, build bounded prompt packs, and optionally call an OpenAI-compatible LLM.

This repo is **Rust-only**.

### Target-agnostic by design

`proofpatch` is intended to work with **any Lean 4 project**:

- you pass a Lean repo root via `--repo /abs/path/to/lean-repo`
- you target a file/lemma/region inside that repo
- you get bounded, structured JSON back (plus optional HTML artifacts for humans)

Any repo-specific wiring (e.g. a `./Scripts/check.sh` in some project) is an **integration example**,
not something `proofpatch` depends on.

### What it does

- **Verify**: run `lake env lean` on a file or on an in-memory snippet.
- **Triage**: count errors/warnings, find nearest `sorry`, and emit a small “what to do next” JSON.
- **Prompt packs**: extract a bounded excerpt + imports + nearby decl headers.
- **Review packs**: build a bounded git-diff context with secret redaction.
- **MCP server**: expose the same operations via MCP (HTTP + optional stdio).

### Quickstart (CLI)

From the repo root:

```bash
cargo run --quiet -p proofpatch-core --bin proofpatch -- --help
```

Common commands (all output JSON):

```bash
# Verify + sorry scan (no LLM)
cargo run --quiet -p proofpatch-core --bin proofpatch -- triage-file \
  --repo /abs/path/to/lean-repo \
  --file Some/File.lean

# Build a bounded context pack around a declaration (no LLM)
cargo run --quiet -p proofpatch-core --bin proofpatch -- context-pack \
  --repo /abs/path/to/lean-repo \
  --file Some/File.lean \
  --decl some_theorem

# Suggest a proof for a lemma (LLM call)
cargo run --quiet -p proofpatch-core --bin proofpatch -- suggest \
  --repo /abs/path/to/lean-repo \
  --file Some/File.lean \
  --lemma some_theorem

# Patch first `sorry` in the lemma using a file and verify (in-memory; does not write)
cargo run --quiet -p proofpatch-core --bin proofpatch -- patch \
  --repo /abs/path/to/lean-repo \
  --file Some/File.lean \
  --lemma some_theorem \
  --replacement-file /tmp/replacement.lean
```

### Review-diff (LLM) prompt bounding

`review-diff` supports explicit size caps:

- `--max-total-bytes N` (overall prompt budget)
- `--per-file-bytes N` (cap per selected file excerpt)
- `--transcript-bytes N` (cap optional transcript tail; default is off)

### Environment (LLM routing)

Prefer `PROOFPATCH_*` env vars.

- **Provider order**: `PROOFPATCH_PROVIDER_ORDER` (default `ollama,groq,openai,openrouter`)
- **Ollama**: `OLLAMA_MODEL` (+ optional `OLLAMA_HOST`)
- **Groq**: `GROQ_API_KEY`, `GROQ_MODEL`
- **OpenAI**: `OPENAI_API_KEY`, `OPENAI_MODEL` (+ optional `OPENAI_BASE_URL`)
- **OpenRouter**: `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` (+ optional `OPENROUTER_BASE_URL`)

Verification behavior:
- **Auto-build**: `PROOFPATCH_AUTO_BUILD=0` disables the “missing olean → lake build → retry” fallback.

Environment loading (super-workspace convenience):
- reads `<repo_root>/.env` if present (does not override already-set vars)
- optionally searches one directory deep for a sibling `.env` if no API key is set yet  
  controls: `PROOFPATCH_DOTENV_SEARCH=0`, `PROOFPATCH_DOTENV_SEARCH_ROOT=/abs/path`

### MCP server

```bash
cargo run --quiet -p proofpatch-mcp --bin proofpatch-mcp
```

Defaults:
- `PROOFPATCH_MCP_ADDR=127.0.0.1:8087`
- `PROOFPATCH_MCP_TOOL_TIMEOUT_S=180`

