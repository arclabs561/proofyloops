# SMT oracle (optional, via smtkit)

`proofyloops` can optionally use an external SMT solver (via `smtkit`) as a **heuristic signal** for small linear integer arithmetic (LIA) entailment checks.

- **Purpose**: rank/prune candidates in `tree-search-nearest` using cheap entailment checks.
- **Soundness**: Lean verification is the only “real” check; SMT is advisory.

## Quickstart: probe solver capabilities

Canonical command:

```bash
proofyloops smt-probe
```

Alias (equivalent):

```bash
proofyloops smt probe
```

This reports:

- the solver command line used (best-effort auto-detect unless `--smt-solver ...` is set)
- a capability matrix (best-effort): `check_sat_assuming`, `get_model`, `get_unsat_core`, `get_proof`

## Enabling SMT inside `tree-search-nearest`

SMT checks usually need goal extraction first. For best results, run with:

```bash
proofyloops tree-search-nearest ... --goal-dump --smt-precheck
```

Useful knobs (all are optional; defaults are conservative):

- **`--smt-solver <auto|...>`**: choose solver backend (or let it auto-detect).
- **`--smt-timeout-ms <n>`**: cap each solver call (default is small).
- **`--smt-depth <n>`**: how much context to include in entailment prechecks.
- **`--smt-seed <n>`**: determinism hook (only used when the solver supports it).
- **`--smt-unsat-core`**, **`--smt-unsat-core-max <n>`**: request/limit UNSAT cores when supported.
- **`--smt-proof`**, **`--smt-proof-max-chars <n>`**: capture a bounded UNSAT proof object when supported.
- **`--smt-proof-dump`**, **`--smt-proof-dump-dir <dir>`**: write proof S-expressions to disk (only when the solver returns a full proof object).
- **`--smt-dump`**, **`--smt-dump-dir <dir>`**, **`--smt-dump-max <n>`**: write bounded `.smt2` scripts to disk for reproduction.
- **`--smt-repro-dir <dir>`**: write a self-contained repro bundle directory (implies `--goal-dump`).

## `smt-repro`: emit a standalone `.smt2` (and optional proof)

Canonical command:

```bash
proofyloops smt-repro --input-json run.json --emit-smt2 repro.smt2 --emit-proof repro.sexp
```

Alias (equivalent):

```bash
proofyloops smt repro --input-json run.json --emit-smt2 repro.smt2 --emit-proof repro.sexp
```

`--input-json` can be either:

- a raw `pp_dump` JSON object, or
- a full `tree-search-nearest` output (it will read `goal_dump.pp_dump`).

## MCP surface

Two equivalent ways to probe solver capabilities:

- **Minimal toolset** (`PROOFYLOOPS_MCP_TOOLSET=minimal`): call the `proofyloops` tool with `action: "smt_probe"`.
- **Full toolset** (`PROOFYLOOPS_MCP_TOOLSET=full`): call the `proofyloops_smt_probe` tool (or the short alias `smt_probe`).

Two equivalent ways to run repro:

- **Minimal toolset** (`PROOFYLOOPS_MCP_TOOLSET=minimal`): call the `proofyloops` tool with `action: "smt_repro"` (aliases like `smt.repro` also work).
- **Full toolset** (`PROOFYLOOPS_MCP_TOOLSET=full`): call the `proofyloops_smt_repro` tool (or the short alias `smt_repro`).

