use serde_json::Value;

/// Best-effort JSON extraction from model output.
///
/// This is intentionally conservative and dependency-free:
/// - prefer ```json fenced blocks
/// - otherwise parse the first complete JSON object or array
///
/// Returns `None` if no valid JSON object/array can be extracted.
pub fn extract_first_json_value(s: &str) -> Option<Value> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    // 1) Prefer fenced ```json ... ``` blocks.
    if let Some(i) = s.find("```json") {
        let rest = &s[i + "```json".len()..];
        if let Some(j) = rest.find("```") {
            let cand = rest[..j].trim();
            if let Ok(v) = serde_json::from_str::<Value>(cand) {
                return Some(v);
            }
        }
    }

    // A model can emit prose or a second value after the first one. Ask the
    // JSON parser where that value ends, including delimiters inside strings.
    s.match_indices(['{', '[']).find_map(|(i, _)| {
        serde_json::Deserializer::from_str(&s[i..])
            .into_iter::<Value>()
            .next()?
            .ok()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extracts_one_complete_value_from_prose() {
        let reply = r#"first {"text":"} and ["} then {"next":2}"#;
        assert_eq!(
            extract_first_json_value(reply),
            Some(json!({"text":"} and ["}))
        );
        assert_eq!(
            extract_first_json_value("Tactics: [\"simp\", \"rfl\"] done"),
            Some(json!(["simp", "rfl"]))
        );
        assert_eq!(
            extract_first_json_value("[note] then {\"x\":1}"),
            Some(json!({"x":1}))
        );
    }

    #[test]
    fn explicit_json_fence_still_has_priority() {
        let reply = "{\"example\":1}\n```json\n[2]\n```";
        assert_eq!(extract_first_json_value(reply), Some(json!([2])));
    }
}
