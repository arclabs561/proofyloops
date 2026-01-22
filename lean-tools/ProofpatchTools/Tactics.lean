import Lean

open Lean Meta Elab Tactic

namespace ProofpatchTools

private def exprToString (e : Expr) : MetaM String := do
  let fmt ← ppExpr e
  pure fmt.pretty

private def localDeclToJson (d : LocalDecl) : MetaM Json := do
  let name := d.userName.toString
  let tyStr ← exprToString d.type
  return Json.mkObj [
    ("name", Json.str name),
    ("type", Json.str tyStr)
  ]

private def goalToJson (g : MVarId) : MetaM Json := do
  g.withContext do
    -- `ppGoal` is the most stable way to get both locals + target across Lean versions.
    let fmt ← Lean.Meta.ppGoal g
    return Json.mkObj [
      ("pretty", Json.str fmt.pretty)
    ]

/-!
`pp_dump` prints the current goals + local context as a single JSON object.

Typical use while iterating on a proof:

```lean
by
  pp_dump
  sorry
```
-/
elab "pp_dump" : tactic => do
  let goals ← getGoals
  let mut goalsJson : Array Json := #[]
  for g in goals do
    goalsJson := goalsJson.push (← liftMetaM (goalToJson g))
  let out := Json.mkObj [
    ("tool", Json.str "proofpatch"),
    ("kind", Json.str "pp_dump"),
    ("goals", Json.arr goalsJson)
  ]
  -- Emit one line (machine-friendly).
  logInfo m!"{toString out}"

end ProofpatchTools

