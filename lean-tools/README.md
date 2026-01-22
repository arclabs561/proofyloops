# proofpatch Lean tools

This is a tiny, dependency-free Lean 4 package that provides helper tactics for `proofpatch`.

## Build

```bash
cd lean-tools
lake build
```

## Use in another Lean repo (non-invasive)

1) Build this package once.
2) Set `PROOFPATCH_EXTRA_LEAN_PATH` to its build output directory:

```bash
export PROOFPATCH_EXTRA_LEAN_PATH="$(pwd)/lean-tools/.lake/build/lib/lean"
```

3) In the target file, add:

```lean
import ProofpatchTools
```

4) Inside a `by` proof, call:

```lean
pp_dump
```

You should see an `info:` log line containing a single-line JSON object describing goals/locals.

