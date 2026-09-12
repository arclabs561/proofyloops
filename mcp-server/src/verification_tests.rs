use super::summarize_verify_like_output as summarize;
use serde_json::json;

#[test]
fn admission_warnings_accept_lean_quote_styles_on_both_streams() {
    // Backtick form reproduced with Lean 4.30.0-rc1 on an incomplete theorem.
    for quote in ['\'', '`'] {
        for token in ["sorry", "admit"] {
            for stream in ["stdout", "stderr"] {
                let mut raw = json!({"ok": true, "returncode": 0, "stdout": "", "stderr": ""});
                raw[stream] = json!(format!(
                    "Example.lean:4:8: warning: declaration uses {quote}{token}{quote}\n"
                ));
                let result = summarize(&raw);
                assert_eq!(result["ok"], true);
                assert_eq!(result["counts"]["sorry_warnings"], 1, "{raw}");
            }
        }
    }
}

#[test]
fn ordinary_warnings_are_not_admissions() {
    let result = summarize(&json!({
        "ok": true, "stdout": "Example.lean:1:1: warning: unused variable `n`\n", "stderr": ""
    }));
    assert_eq!(result["counts"]["sorry_warnings"], 0);
}
