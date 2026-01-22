use proofpatch_core::tree_search as ts;

#[test]
fn sanitize_candidates_dedupes_and_bounds() {
    let xs = vec![
        "".to_string(),
        "by\n  simp".to_string(),
        "by\n  simp".to_string(),
        "  ".to_string(),
    ];
    let out = ts::sanitize_candidates(xs);
    assert_eq!(out, vec!["by\n  simp".to_string()]);
}

#[test]
fn parse_json_string_array_accepts_strings_only() {
    let s = r#"["by\n  simp","by\n  aesop"]"#;
    let out = ts::parse_json_string_array(s).unwrap();
    assert_eq!(out.len(), 2);
    assert!(out[0].contains("simp"));
}

#[test]
fn parse_json_string_array_can_extract_from_markdown_fence() {
    let s = "Here you go:\n```json\n[\"a\",\"b\"]\n```\n";
    let out = ts::parse_json_string_array(s).unwrap();
    assert_eq!(out, vec!["a".to_string(), "b".to_string()]);
}

#[test]
fn made_no_progress_detection() {
    assert!(ts::is_made_no_progress(Some("tactic 'simp' made no progress")));
    assert!(!ts::is_made_no_progress(Some("unknown constant")));
}

#[test]
fn adapt_candidates_for_sorry_line_strips_by_prefix() {
    let base = vec!["by\n  simp".to_string(), "by aesop".to_string()];
    let out = ts::adapt_candidates_for_sorry_line(&base, "  by sorry");
    assert_eq!(out, vec!["simp".to_string(), "aesop".to_string()]);
}

#[test]
fn adapt_candidates_for_sorry_line_does_not_strip_on_bare_sorry() {
    let base = vec!["by\n  simp".to_string(), "by aesop".to_string()];
    let out = ts::adapt_candidates_for_sorry_line(&base, "  sorry");
    assert_eq!(out, base);
}

#[test]
fn adapt_candidates_for_sorry_context_strips_when_tactic_context() {
    let base = vec!["by\n  simp".to_string(), "by aesop".to_string()];
    let out = ts::adapt_candidates_for_sorry_context(&base, "  sorry", true);
    assert_eq!(out, vec!["(simp; done)".to_string(), "(aesop; done)".to_string()]);
}

#[test]
fn extract_initial_goal_block_finds_block() {
    let s = r#"
error: tactic 'aesop' failed, made no progress
Initial goal:
  x : Nat
  ⊢ True

Foo.lean:1:2: error: boom
"#;
    let b = ts::extract_initial_goal_block(s).expect("expected block");
    assert!(b.contains("Initial goal:"));
    assert!(b.contains("⊢ True"));
}

