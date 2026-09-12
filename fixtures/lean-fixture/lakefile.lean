import Lake

open Lake DSL

package "lean-fixture" where
  -- keep it tiny; this is a CI/e2e fixture for proofyloops only

@[default_target]
lean_lib LeanFixture where
  roots := #[`LeanFixture]
