use proofyloops_core::config;

#[test]
fn research_preset_accepts_must_include_all() {
    let txt = r#"
[research.presets.demo]
query = "LLL swap count"
must_include_any = ["lll"]
must_include_all = ["swap", "count"]
"#;
    let cfg: config::ProofyloopsConfig = toml::from_str(txt).expect("toml parse");
    let p = cfg.research.resolve_preset("demo").expect("preset");
    assert_eq!(p.query, "LLL swap count");
    assert_eq!(p.must_include_any, vec!["lll"]);
    assert_eq!(p.must_include_all, vec!["swap", "count"]);
}

#[test]
fn research_preset_merges_tree_search_defaults() {
    let txt = r#"
[research.defaults.tree_search]
goal_first_k = 5
smt_depth = 2
smt_solver = "z3"
smt_timeout_ms = 2500
smt_explain = true
smt_explain_max_hyps = 9
hint_packs = ["base"]

[research.presets.demo]
query = "LLL swap count"

[research.presets.demo.tree_search]
goal_first_k = 7
smt_solver = "cvc5"
smt_timeout_ms = 4000
hint_packs = ["demo"]
"#;
    let cfg: config::ProofyloopsConfig = toml::from_str(txt).expect("toml parse");
    let p = cfg.research.resolve_preset("demo").expect("preset");
    let ts = p.tree_search.expect("tree_search");
    assert_eq!(ts.goal_first_k, Some(7));
    assert_eq!(ts.smt_depth, Some(2));
    assert_eq!(ts.smt_solver.as_deref(), Some("cvc5"));
    assert_eq!(ts.smt_timeout_ms, Some(4000));
    assert_eq!(ts.smt_explain, Some(true));
    assert_eq!(ts.smt_explain_max_hyps, Some(9));
    let expected = vec!["demo".to_string()];
    assert_eq!(ts.hint_packs.as_deref(), Some(expected.as_slice()));
}

#[test]
fn repo_config_prefers_proofyloops_name_over_legacy_name() {
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::write(
        dir.path().join("proofyloops.toml"),
        "[research.presets.canonical]\nquery = \"canonical\"\n",
    )
    .expect("canonical config");
    std::fs::write(
        dir.path().join("proofpatch.toml"),
        "[research.presets.legacy]\nquery = \"legacy\"\n",
    )
    .expect("legacy config");

    let cfg = config::load_from_repo_root(dir.path())
        .expect("canonical config parses")
        .expect("config exists");
    assert!(cfg.research.presets.contains_key("canonical"));
    assert!(!cfg.research.presets.contains_key("legacy"));
}

#[test]
fn repo_config_uses_legacy_name_when_canonical_is_absent() {
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::write(
        dir.path().join("proofpatch.toml"),
        "[research.presets.legacy]\nquery = \"legacy\"\n",
    )
    .expect("legacy config");

    let cfg = config::load_from_repo_root(dir.path())
        .expect("legacy config parses")
        .expect("config exists");
    assert!(cfg.research.presets.contains_key("legacy"));
}

#[test]
fn invalid_canonical_config_does_not_fall_back_to_legacy_name() {
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::write(dir.path().join("proofyloops.toml"), "[research\n")
        .expect("invalid canonical config");
    std::fs::write(
        dir.path().join("proofpatch.toml"),
        "[research.presets.legacy]\nquery = \"legacy\"\n",
    )
    .expect("legacy config");

    let error = config::load_from_repo_root(dir.path()).expect_err("canonical error must win");
    assert!(error.contains("proofyloops.toml"));
}

#[test]
fn research_suggestion_uses_longest_matching_declaration_selector() {
    let txt = r#"
[research.presets.short]
query = "short query"
when_decl_contains = ["sum_three"]

[research.presets.long]
query = "long query"
when_decl_contains = ["sum_three_squares"]
"#;
    let cfg: config::ProofyloopsConfig = toml::from_str(txt).expect("toml parse");

    assert_eq!(
        cfg.research
            .suggestion_for_decl("sum_three_squares_iff")
            .preset_name,
        Some("long".to_string())
    );
    assert_eq!(
        cfg.research
            .suggestion_for_decl("sum_three_squares_iff")
            .query,
        "long query"
    );
}

#[test]
fn research_suggestion_breaks_equal_selector_ties_by_preset_name() {
    let txt = r#"
[research.presets.zeta]
query = "zeta query"
when_decl_contains = ["lemma"]

[research.presets.alpha]
query = "alpha query"
when_decl_contains = ["lemma"]
"#;
    let cfg: config::ProofyloopsConfig = toml::from_str(txt).expect("toml parse");

    let suggestion = cfg.research.suggestion_for_decl("some_lemma");
    assert_eq!(suggestion.preset_name.as_deref(), Some("alpha"));
    assert_eq!(suggestion.query, "alpha query");
}

#[test]
fn research_suggestion_ignores_empty_selectors_and_has_generic_fallback() {
    let txt = r#"
[research.presets.empty]
query = "should not select"
when_decl_contains = ["", "   "]
"#;
    let cfg: config::ProofyloopsConfig = toml::from_str(txt).expect("toml parse");

    let suggestion = cfg.research.suggestion_for_decl("ordinary_lemma");
    assert_eq!(suggestion.preset_name, None);
    assert_eq!(suggestion.query, "Lean proof ordinary_lemma");
}

#[test]
fn rubberduck_plan_only_names_a_research_preset_when_one_matches() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("lakefile.lean"), "package demo\n").unwrap();
    std::fs::write(dir.path().join("lean-toolchain"), "v4.0.0\n").unwrap();
    std::fs::write(
        dir.path().join("Example.lean"),
        "theorem selected_lemma : True := by trivial\ntheorem other_lemma : True := by trivial\n",
    )
    .unwrap();
    std::fs::write(
        dir.path().join("proofyloops.toml"),
        "[research.presets.demo]\nquery = \"custom research\"\nwhen_decl_contains = [\"selected_lemma\"]\n",
    )
    .unwrap();

    for (decl, expected_query, expected_preset) in [
        (
            "selected_lemma",
            "custom research mathlib Lean",
            Some("demo"),
        ),
        ("other_lemma", "Lean proof other_lemma mathlib Lean", None),
    ] {
        let prompt =
            proofyloops_core::build_rubberduck_prompt(dir.path(), "Example.lean", decl, None)
                .unwrap();
        let (_, raw_plan) = prompt
            .user
            .rsplit_once("- Suggested research plan (machine-readable JSON):\n")
            .unwrap();
        let plan: serde_json::Value = serde_json::from_str(raw_plan.trim()).unwrap();
        let steps = plan["steps"].as_array().unwrap();
        let web = steps.last().unwrap();
        assert_eq!(web["tool"], "web_search");
        assert_eq!(web["args"]["query"], expected_query);
        if let Some(preset) = expected_preset {
            assert_eq!(steps.len(), 2);
            assert_eq!(steps[0]["tool"], "proofyloops research-auto");
            assert_eq!(steps[0]["args"]["preset"], preset);
        } else {
            assert_eq!(steps.len(), 1);
        }
    }
}
