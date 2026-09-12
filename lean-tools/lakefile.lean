import Lake
open Lake DSL

package proofyloops_lean_tools where
  -- Keep this package dependency-free (Lean core only).

@[default_target]
lean_lib ProofyloopsTools where
  roots := #[`ProofyloopsTools]

