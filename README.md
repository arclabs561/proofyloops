# proofpatch

`proofpatch` is a CLI + MCP server for **debuggable Lean 4 workflows**: verify, locate `sorry`s, extract bounded context packs, and (optionally) call an OpenAI-compatible LLM.

### Design

- **Target-agnostic**: point at a Lean repo with `--repo`, then target a file/decl/region inside it.
- **Evidence-first**: commands return structured JSON and can emit on-disk artifacts under `.generated/`.

### Quickstart (CLI)

From this repo:

```bash
cargo run -p proofpatch-cli --bin proofpatch -- --help
```

Typical command:

```bash
cargo run -p proofpatch-cli --bin proofpatch -- triage-file \
  --repo /abs/path/to/lean-repo \
  --file Some/File.lean
```

### MCP server

```bash
cargo run -p proofpatch-mcp --bin proofpatch-mcp
```

### SMT (optional oracle)

`proofpatch` can use `smtkit` as a **heuristic signal** (never as a proof) for LIA entailment checks.

### Quickstart: `tree-search-nearest` with SMT evidence

If you have an SMT solver on PATH (e.g. `z3`), a good “debuggable” run is:

```bash
proofpatch smt-probe

proofpatch tree-search-nearest \
  --repo /abs/path/to/lean-repo \
  --file Some/File.lean \
  --goal-dump \
  --smt-precheck \
  --smt-dump --smt-dump-dir .generated/proofpatch-smt2 \
  --smt-repro-dir .generated/proofpatch-smtrepro \
  --output-json .generated/proofpatch-tree-search.json
```

- `--goal-dump` + `--smt-precheck` makes SMT checks much more informative (it extracts hypotheses/targets).
- `--smt-dump` writes bounded `.smt2` scripts to disk for inspection.
- `--smt-repro-dir` writes a self-contained repro bundle you can re-run with `proofpatch smt-repro`.

See `docs/smt.md` for:
- how dumping/repro bundles work
- UNSAT core / proof capture knobs
- `smt-repro` usage

### More docs

- `docs/usage.md`: common CLI patterns, focus flags, output stability.
- `lean-tools/README.md`: `ProofpatchTools` (Lean side helper tactics).
- `proofpatch-lean-embed/README.md`: optional Lean runtime embedding.
