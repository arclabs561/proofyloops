import Lake
open Lake DSL

package proofpatch_lean_tools where
  -- Keep this package dependency-free (Lean core only).

@[default_target]
lean_lib ProofpatchTools where
  roots := #[`ProofpatchTools]

