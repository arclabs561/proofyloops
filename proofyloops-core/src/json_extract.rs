use serde_json::Value;

/// Best-effort JSON extraction from model output.
///
/// This is intentionally conservative and dependency-free:
/// - prefer ```json fenced blocks
/// - otherwise parse the first `{ ... }` span
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

    // 2) Fall back to parsing the first {...} span.
    let i = s.find('{')?;
    let j = s.rfind('}')?;
    if j <= i {
        return None;
    }
    let cand = s[i..=j].trim();
    serde_json::from_str::<Value>(cand).ok()
}
