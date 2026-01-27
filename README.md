## proofpatch

[![CI](https://github.com/arclabs561/proofpatch/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/proofpatch/actions/workflows/ci.yml)

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

### SMT integration (motivated + auditable)

`proofpatch` can optionally use an external SMT solver as a **heuristic signal** (never as a proof):

- **Purpose**: rank/prune candidates in `tree-search-nearest` using cheap LIA entailment checks.
- **Soundness**: Lean verification is the only “real” check; SMT is advisory.

When SMT is enabled, `tree-search-nearest` records enough evidence to debug and reproduce behavior:

- **Solver capability matrix**: included under `oracle.smt.solver.caps` (best-effort),
  including whether the current solver supports:
  - `check_sat_assuming`
  - `get_unsat_core`
  - `get_proof` (solver-dependent; captured as a provenance/debug hook)
- **SMT2 artifacts**:
  - `oracle.smt.dump_paths` points at `.smt2` scripts written to disk (when `--smt-dump` is set).
  - `oracle.smt.dump_last.preview` includes a bounded inline preview of the last emitted script.

### Dependency pinning (recommended)

- `proofpatch` treats SMT as an *optional oracle* via `smtkit`.
- For reproducibility, prefer the published crates (`smtkit = "0.x.y"`) and commit `Cargo.lock` (this repo does).
- For testing unreleased `smtkit` changes, temporarily pin a git tag/rev, then revert to crates.io on release.

If you want to capture an UNSAT proof object (solver-dependent), enable:

```bash
proofpatch tree-search-nearest ... --smt-proof --smt-proof-max-chars 12000
```

The output includes a bounded preview under `oracle.smt.proofs.last` and (when an entailment check succeeds)
also under the per-node `smt_hint.unsat_proof`.

If you want a reproducible on-disk artifact (so you can inspect the full-ish proof object without digging through JSON),
add:

```bash
proofpatch tree-search-nearest ... --smt-proof-dump --smt-proof-dump-dir .generated/proofpatch-smtproof
```

Notes:
- **Flag behavior**: `--smt-proof-dump` implies `--smt-proof`.
- **When dumps are written**: proof dumps are only written when the SMT entailment check returns `entails=true` and the solver produces an UNSAT proof object via `(get-proof)`.
- **Size cap behavior**: if the proof is larger than `--smt-proof-dump-max-chars`, `proofpatch` will **skip writing** the `.sexp` file (to avoid writing a likely-invalid truncated S-expression) and will record the reason under `oracle.smt.proofs.dump.last_error`.
- **File format**: the dump file is a single S-expression (typically a `(proof ...)` term), saved as `.sexp`.

Practical note: SMT dumping typically needs goal dumps to learn hypotheses/targets. If you want dumps, run:

```bash
proofpatch tree-search-nearest ... --goal-dump --smt-precheck --smt-dump
```

Similarly, SMT proof dumping is most reliable with:

```bash
proofpatch tree-search-nearest ... --goal-dump --smt-precheck --smt-proof-dump
```

If you want to reproduce an SMT entailment/proof *outside* of `tree-search-nearest`, you can use:

```bash
proofpatch smt-repro \
  --input-json run.json \
  --emit-smt2 repro.smt2 \
  --emit-proof repro.sexp
```

`--input-json` can be either:
- a raw `pp_dump` JSON object, or
- a full `tree-search-nearest` JSON output (it will read `goal_dump.pp_dump`).

You can also generate a standalone `pp_dump` JSON (for feeding into `smt-repro`) with:

```bash
proofpatch goal-dump-nearest --repo /abs/path/to/lean-repo --file Some/File.lean --output-json goal_dump.json
```

This requires the file/text to contain at least one `sorry`/`admit` token (it patches one temporarily).

For a one-shot pipe (no intermediate file), use:

```bash
proofpatch goal-dump-nearest --repo /abs/path/to/lean-repo --file Some/File.lean --pp-dump-only \
  | proofpatch smt-repro --input-json - --emit-smt2 repro.smt2 --emit-proof repro.sexp
```

If the target file/decl is **sorry-free**, you can still extract a `pp_dump` by synthesizing a
temporary “shadow decl” with the same signature (either by naming the decl, or by giving a focus line near it):

```bash
proofpatch goal-dump-nearest --repo /abs/path/to/lean-repo --file Some/File.lean \
  --focus-decl MyDecl --allow-sorry-free --pp-dump-only \
  | proofpatch smt-repro --input-json - --emit-smt2 repro.smt2 --emit-proof repro.sexp
```

Or:

```bash
proofpatch goal-dump-nearest --repo /abs/path/to/lean-repo --file Some/File.lean \
  --focus-line 123 --allow-sorry-free --pp-dump-only \
  | proofpatch smt-repro --input-json - --emit-smt2 repro.smt2 --emit-proof repro.sexp
```

Note: `smt-repro` only emits `repro.smt2` when it can recognize the goal as a supported SMT-LIA entailment; otherwise it returns `result_kind: "no_entailment"` and leaves `smt2_written` empty.

You can also write a portable “goal capsule” bundle (goal dump + pp_dump + manifest) with:

```bash
proofpatch goal-dump-nearest --repo /abs/path/to/lean-repo --file Some/File.lean \
  --focus-decl MyDecl --allow-sorry-free --bundle-dir goal_capsule/
```

To speed up the inner loop, you can run a bounded “tactic try” pass over a capsule:

```bash
proofpatch goal-try --capsule-dir goal_capsule/ --top-k 8 --timeout-s 25 --rounds 2 --beam 2
```

This executes a small ranked list of deterministic tactics (derived from `pp_dump`) against the
generated `shadow.lean` and:
- reports the first tactic that fully closes the goal (if any), or
- when nothing closes the goal, returns a **best-effort “progress best”** attempt ranked by a small
  score derived from the post-tactic `pp_dump` (remaining goals + total goal text size + hypothesis counts).

Optional flags:
- `--with-try-this`: also run mathlib suggestion tactics (e.g. `simp?`, `aesop?`) to augment the candidate list.
- `--write-best`: write `shadow.best.lean` into the capsule directory (either the solved proof, or the best progress attempt).
- `--rounds N`: run up to N rounds of “best progress” hill-climbing (each round applies the best progress attempt and tries again).
- `--beam N`: keep the top N improving branches per round (bounded beam search; default 1).

For a portable “capsule” directory (inputs + outputs + manifest), use `--bundle-dir`:

```bash
proofpatch smt-repro --input-json goal_dump.json --bundle-dir smt_capsule/
```

This writes:
- `smt_capsule/pp_dump.json`
- `smt_capsule/smt_repro.json`
- `smt_capsule/manifest.json` (hashes + parameters + solver probe)
- and (when available) `smt_capsule/repro.smt2`, `smt_capsule/repro.sexp`

If you want `tree-search-nearest` to auto-produce a self-contained repro bundle (including the full
`tree_search.json`, plus `repro.smt2` / `repro.sexp` / `smt_repro.json`), pass:

```bash
proofpatch tree-search-nearest ... --smt-repro-dir .generated/proofpatch-smtrepro
```

If the target directory already contains a previous bundle, `proofpatch` will write into a fresh
`run_<timestamp>/` subdirectory to avoid overwriting.

Notes:
- `--smt-repro-dir` implies `--goal-dump`.
- `--smt-repro-dir` also enables SMT proof capture (so `oracle.smt.proofs` is populated in the main JSON when possible).

### What works well (and what doesn’t)

- **Best workflow for “real progress” in a Lean repo**:
 - Create a temporary scratch file under `<repo_root>/.generated/` with *one* targeted lemma.
 - Run `proofpatch tree-search-nearest` with a repo-owned `--research-preset` (so goal-dump + SMT precheck are enabled and you get bounded context).
 - Once the proof compiles in scratch, **port the proof into the real file** and delete the scratch file so the repo stays `sorry`-free.

- **When `tree-search-nearest` helps a lot**:
 - “Local” goals: small `simp`/`ring_nf` steps, closing a leaf inequality, finding a missing lemma name, or testing whether a refactor broke a proof.
 - Quickly validating that a “textbook” lemma can be derived from an existing canonical lemma by rewriting.

- **When it tends not to help** (yet):
 - Large combinatorial re-indexing proofs or “global” algebraic reorganizations. In practice, these usually want a human-authored proof skeleton plus a few targeted lemma lookups.
 - Anything where the important move is selecting the *right representation* (e.g. switch from a \(\Phi\)-style product to an order-invariant \(\Psi\)-style product); the tool can validate the end state, but it won’t reliably discover the representation change.

- **Reports and artifacts**:
 - `tree-search-nearest` writes optional artifacts under `<repo_root>/.generated/proofpatch-cache/…` by default (when caching is enabled).
 - The JSON output now includes a bounded `artifacts.report_md.preview`, so you can read the human report even if the report path is inconvenient to open in your editor.
 - If the target file is **already sorry-free**, `tree-search-nearest` returns immediately with `ok=true` and a short `note` (no expensive search).

### Quickstart (CLI)

From the repo root:

```bash
cargo run --quiet -p proofpatch-cli --bin proofpatch -- --help
```

Common commands (all output JSON):

```bash
# Verify + sorry scan (no LLM)
cargo run --quiet -p proofpatch-cli --bin proofpatch -- triage-file \
  --repo /abs/path/to/lean-repo \
  --file Some/File.lean

# Build a bounded context pack around a declaration (no LLM)
cargo run --quiet -p proofpatch-cli --bin proofpatch -- context-pack \
  --repo /abs/path/to/lean-repo \
  --file Some/File.lean \
  --decl some_theorem

# Suggest a proof for a lemma (LLM call)
cargo run --quiet -p proofpatch-cli --bin proofpatch -- suggest \
  --repo /abs/path/to/lean-repo \
  --file Some/File.lean \
  --lemma some_theorem

# Patch first `sorry` in the lemma using a file and verify (in-memory; does not write)
cargo run --quiet -p proofpatch-cli --bin proofpatch -- patch \
  --repo /abs/path/to/lean-repo \
  --file Some/File.lean \
  --lemma some_theorem \
  --replacement-file /tmp/replacement.lean
```

Tip: `proofpatch` can also be installed as a normal CLI binary (so you can run `proofpatch ...` directly):

```bash
cargo install --path proofpatch-cli --bin proofpatch --force
```

### Research loop (repo-owned presets)

If your Lean repo contains a `proofpatch.toml`, you can run a bounded research fetch that produces a
reproducible JSON artifact:

```bash
proofpatch research-auto --repo /abs/path/to/lean-repo --preset <name> --output-json /tmp/research.json
```

Notes:
- If arXiv rate-limits (`429`), `research-auto` uses a small bounded retry/backoff.
 - You can cap retries with `PROOFPATCH_ARXIV_MAX_RETRIES=0` (or another small integer).

Then you can use that preset to condition `tree-search-nearest` (so research becomes an active input
to the exploration loop):

```bash
proofpatch tree-search-nearest --repo /abs/path/to/lean-repo --file Some/File.lean --research-preset <name> --output-json /tmp/run.json
```

### Scratch workflow helper

To speed up “work backwards” iterations, `scratch-lemma` creates a `.generated/...` file with the right
`import` and a placeholder `sorry`:

```bash
proofpatch scratch-lemma --repo /abs/path/to/lean-repo --file Some/File.lean --name my_lemma --out .generated/my_lemma.lean
```

### Targeting a specific lemma (focus)

If a file has multiple `sorry`s, you can “pin” the search to a specific declaration:

```bash
proofpatch tree-search-nearest --repo /abs/path/to/lean-repo --file Some/File.lean --focus-decl MyNamespace.my_lemma
```

If you want to **avoid drifting** to other declarations even if the focused decl has no more `sorry`s:

```bash
proofpatch tree-search-nearest --repo /abs/path/to/lean-repo --file Some/File.lean --focus-decl MyNamespace.my_lemma --focus-decl-hard
```

If you want `proofpatch` to **fail fast** when the decl name doesn’t match any `sorry` location:

```bash
proofpatch tree-search-nearest --repo /abs/path/to/lean-repo --file Some/File.lean --focus-decl MyNamespace.my_lemma --focus-decl-strict
```

### Output stability: `result_kind`

Many commands can early-exit (e.g. “file has no `sorry`”, “hard focus mismatch”). To make downstream
parsing reliable, outputs include a stable `result_kind` string such as:

- `early_no_sorries`
- `early_focus_decl_strict_not_found`
- `early_focus_decl_strict_empty`
- `early_focus_decl_hard_not_found`
- `early_focus_decl_hard_empty`
- `search`
- `solved` (goal-try)
- `progress` (goal-try)
- `unsolved` (goal-try)

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

### CI / E2E testing

This repo includes a tiny Lean project fixture under `fixtures/lean-fixture/` used by CI to exercise:

- `proofpatch triage-file` against a real Lake project
- the stdio MCP surface via `mcp-server/examples/stdio_smoke.rs`

### Agentic tool-calling mode (context)

The “agent loop + tool-calling contracts” engine lives in [`axi`](https://github.com/arclabs561/axi).
`proofpatch` is designed to stay useful without it; any deeper agentic integration should remain **optional**.

