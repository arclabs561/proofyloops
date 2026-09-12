use proofyloops_core::{config, derive_candidates_from_goal_pretty_with_hint_rules};

#[test]
fn hint_rule_rewrites_mod8_binder_name() {
    let rule = config::HintRule {
        when_contains_all: vec!["⊢ Odd".to_string(), "% 8 =".to_string()],
        when_contains_any: vec![],
        candidates: vec![
            "by\n  have h2 : n % 2 = 1 := by omega\n  exact Nat.odd_iff.2 h2".to_string(),
        ],
    };

    // Binder name is `m`, not `n`, but the hint rule candidate uses `n`.
    let goal_pretty = r#"
m : Nat
hm : m % 8 = 3
⊢ Odd m
"#;

    let out = derive_candidates_from_goal_pretty_with_hint_rules(goal_pretty, &[rule]);
    assert!(
        out.iter().any(|c| c.contains("m % 2 = 1")),
        "expected rewritten candidate to refer to `m`: {out:?}"
    );
    assert!(
        out.iter().all(|c| !c.contains("n % 2 = 1")),
        "expected not to leave the hardcoded `n` binder behind: {out:?}"
    );
}
