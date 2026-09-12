use proofyloops_core::tree_search as ts;
use proofyloops_core::{config, derive_candidates_from_goal_pretty_with_hint_rules};

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
    assert!(ts::is_made_no_progress(Some(
        "tactic 'simp' made no progress"
    )));
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
    assert_eq!(
        out,
        vec!["(simp; done)".to_string(), "(aesop; done)".to_string()]
    );
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

#[test]
fn replace_in_region_once_only_rewrites_inside_region() {
    let src = "theorem t : True := by\n  exact?\n\n-- another exact?\n";
    let out = ts::replace_in_region_once(src, 1, 2, "exact?", "exact True.intro")
        .expect("expected replacement");
    assert!(out.contains("exact True.intro"));
    assert!(out.contains("-- another exact?"));
}

#[test]
fn replace_in_region_once_supports_multiline_new_text() {
    let src = "theorem t : True := by\n  exact?\n";
    let new_block = "refine True.intro\n  trivial";
    let out =
        ts::replace_in_region_once(src, 1, 2, "exact?", new_block).expect("expected replacement");
    assert!(out.contains("refine True.intro"));
    assert!(out.contains("trivial"));
}

#[test]
fn replace_in_region_first_rewrites_first_occurrence_only() {
    let src = "theorem t : True := by\n  exact?\n  exact?\n";
    let out = ts::replace_in_region_first(src, 1, 3, "exact?", "exact True.intro")
        .expect("expected replacement");
    // First one replaced, second remains.
    assert!(out.contains("exact True.intro"));
    assert_eq!(out.matches("exact?").count(), 1);
}

#[test]
fn hint_rules_can_rewrite_mod8_binder_name() {
    let goal_pretty = r#"
  n1 : Nat
  h : m % 8 = 3
  ⊢ Odd m
"#;
    let rules = vec![config::HintRule {
        when_contains_all: vec!["⊢ Odd".to_string(), "% 8 =".to_string()],
        when_contains_any: vec![],
        candidates: vec![
            "by\n  have h2 : n % 2 = 1 := by omega\n  exact Nat.odd_iff.2 h2".to_string(),
        ],
    }];
    let out = derive_candidates_from_goal_pretty_with_hint_rules(goal_pretty, &rules);
    assert!(out.iter().any(|c| c.contains("have h2 : m % 2 = 1")));
    // Ensure we didn't keep the brittle binder.
    assert!(!out.iter().any(|c| c.contains("have h2 : n % 2 = 1")));
}
