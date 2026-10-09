use proofyloops_core as plc;

#[test]
fn patch_decl_inline_def_replaces_token_only() {
    let src = "def double (n : Nat) : Nat := sorry\n";
    let out = plc::patch_first_sorry_in_decl(src, "double", "n + n").unwrap();
    assert_eq!(out.line, 1);
    assert_eq!(out.before.trim_end(), "def double (n : Nat) : Nat := sorry");
    assert_eq!(out.after.trim_end(), "def double (n : Nat) : Nat := n + n");
    assert!(out.text.contains(":= n + n"));
}

#[test]
fn patch_decl_replaces_admit_too() {
    let src = "theorem t : True := by admit\n";
    let out = plc::patch_first_sorry_in_decl(src, "t", "by\n  trivial").unwrap();
    assert_eq!(out.line, 1);
    assert!(out.before.contains("admit"));
    assert!(out.after.contains("trivial"));
    assert!(!out.after.contains("admit"));
}

#[test]
fn patch_decl_pure_bind_by_block_normalizes_by_prefix() {
    // Note: `patch_first_sorry_in_decl` targets *named* declarations.
    // Anonymous `instance : ...` has no name to match, so we use a named instance here.
    let src = "instance lawfulOption : LawfulMonad Option where\n  pure_bind := by sorry\n";
    let out = plc::patch_first_sorry_in_decl(src, "lawfulOption", "by\n  simp").unwrap();
    // We splice into `... := by sorry`, so replacement becomes `... := by simp`.
    assert!(out.after.contains("pure_bind := by simp"));
    assert!(!out.after.contains("by by"));
}

#[test]
fn patch_decl_stops_at_next_declaration() {
    // `proved` has no sorry; the `sorry` belongs to `other` and must stay.
    let src = "theorem proved : True := by\n  trivial\n\ntheorem other : True := by\n  sorry\n";
    let err = plc::patch_first_sorry_in_decl(src, "proved", "by\n  simp").unwrap_err();
    assert!(err.contains("proved"), "{err}");
    assert!(!plc::decl_block_contains_sorry(src, "proved").unwrap());
    assert!(plc::decl_block_contains_sorry(src, "other").unwrap());
}

#[test]
fn patch_decl_matches_whole_identifier() {
    // `foo` must not match the header of `foo'`, and `foo'` must be findable.
    let src = "theorem foo' : True := sorry\ntheorem foo : True := sorry\n";
    let out = plc::patch_first_sorry_in_decl(src, "foo", "trivial").unwrap();
    assert_eq!(out.line, 2);
    assert_eq!(out.after.trim_end(), "theorem foo : True := trivial");
    let out = plc::patch_first_sorry_in_decl(src, "foo'", "trivial").unwrap();
    assert_eq!(out.line, 1);
    assert_eq!(out.after.trim_end(), "theorem foo' : True := trivial");
}

#[test]
fn patch_region_replaces_token_in_field_line() {
    let src =
        "instance : LawfulMonad Option where\n  pure_bind := by sorry\n  bind_assoc := by sorry\n";
    let out = plc::patch_first_sorry_in_region(src, 2, 2, "by\n  simp").unwrap();
    assert_eq!(out.line, 2);
    assert_eq!(out.before.trim_end(), "  pure_bind := by sorry");
    assert_eq!(out.after.trim_end(), "  pure_bind := by simp");
}

#[test]
fn locate_sorries_finds_basic_cases() {
    let src = "-- comment sorry\n\
def s1 : String := \"sorry\"\n\
def s2 : Nat := 1 -- sorry\n\
def a : Nat := sorry\n\
instance lawfulOption : LawfulMonad Option where\n\
  pure_bind := by sorry\n\
theorem t : True := by admit\n";
    let locs = plc::locate_sorries_in_text(src, 10, 0).unwrap();
    // skip comment-only line, string literal, and trailing line comment; expect three matches
    assert_eq!(locs.len(), 3);
    assert_eq!(locs[0].line, 4);
    assert_eq!(locs[0].token, "sorry");
    assert!(locs[0].line_text.contains("def a"));
    assert_eq!(locs[1].line, 6);
    assert_eq!(locs[1].token, "sorry");
    assert!(locs[1].line_text.contains("pure_bind"));
    assert_eq!(locs[2].line, 7);
    assert_eq!(locs[2].token, "admit");
    assert!(locs[2].line_text.contains("theorem t"));
}

#[test]
fn locate_sorries_ignores_multiline_block_comments() {
    let src = "/-\n\
this block comment mentions sorry\n\
and another line\n\
-/\n\
def a : Nat := sorry\n\
/- nested /- sorry -/ still comment -/\n\
def b : Nat := sorry\n";
    let locs = plc::locate_sorries_in_text(src, 10, 0).unwrap();
    assert_eq!(locs.len(), 2);
    assert_eq!(locs[0].token, "sorry");
    assert_eq!(locs[1].token, "sorry");
    assert!(locs[0].line_text.contains("def a"));
    assert!(locs[1].line_text.contains("def b"));
}

#[test]
fn extract_try_this_suggestions_parses_inline_form() {
    let out = r#"
some log noise
Try this: exact foo a b
more noise
"#;
    let xs = plc::extract_try_this_suggestions(out);
    assert_eq!(xs, vec!["exact foo a b".to_string()]);
}

#[test]
fn extract_try_this_suggestions_parses_block_form_with_tag() {
    let out = r#"
Try this:
  [apply] refine bar ?_
another line
"#;
    let xs = plc::extract_try_this_suggestions(out);
    assert_eq!(xs, vec!["refine bar ?_".to_string()]);
}

#[test]
fn extract_try_this_suggestions_parses_multiline_block() {
    let out = r#"
Try this:
  refine And.intro ?_ ?_
  · exact ha
  · exact hb
unrelated
"#;
    let xs = plc::extract_try_this_suggestions(out);
    assert_eq!(
        xs,
        vec![["refine And.intro ?_ ?_", "· exact ha", "· exact hb"].join("\n")]
    );
}
