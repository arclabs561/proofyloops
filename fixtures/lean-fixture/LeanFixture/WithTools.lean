import ProofyloopsTools

namespace LeanFixture

-- Ensure proofyloops can extend LEAN_PATH to import the helper tools package.
theorem proofyloops_tools_smoke : True := by
  pp_dump
  trivial

end LeanFixture

