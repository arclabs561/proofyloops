namespace LeanEmbedDemo

-- A tiny exported function we can use as a runtime-embedding smoke test.
@[export pp_add_u64]
def addU64 (a b : UInt64) : UInt64 :=
  a + b

end LeanEmbedDemo

