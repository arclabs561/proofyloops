//! Compatibility-aware environment lookup for Proofyloops controls.
//!
//! `PROOFYLOOPS_*` is canonical. During the rename, an unset canonical control
//! may inherit the corresponding `PROOFPATCH_*` value. A present canonical
//! value, including an empty or non-Unicode one, always wins.

use std::env::VarError;

/// Read a Proofyloops environment control, falling back to its legacy name.
///
/// Names outside the `PROOFYLOOPS_*` namespace are read unchanged.
pub fn var(name: &str) -> Result<String, VarError> {
    var_with(name, |key| std::env::var(key))
}

fn var_with<F>(name: &str, lookup: F) -> Result<String, VarError>
where
    F: Fn(&str) -> Result<String, VarError>,
{
    match lookup(name) {
        Err(VarError::NotPresent) => match legacy_name(name) {
            Some(legacy) => lookup(&legacy),
            None => Err(VarError::NotPresent),
        },
        value => value,
    }
}

fn legacy_name(name: &str) -> Option<String> {
    name.strip_prefix("PROOFYLOOPS_")
        .map(|suffix| format!("PROOFPATCH_{suffix}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn lookup(
        values: HashMap<&'static str, Result<&'static str, VarError>>,
    ) -> impl Fn(&str) -> Result<String, VarError> {
        move |name| {
            values
                .get(name)
                .map(|value| value.clone().map(str::to_string))
                .unwrap_or(Err(VarError::NotPresent))
        }
    }

    #[test]
    fn canonical_value_wins_over_legacy_value() {
        let values = HashMap::from([
            ("PROOFYLOOPS_AUTO_BUILD", Ok("0")),
            ("PROOFPATCH_AUTO_BUILD", Ok("1")),
        ]);
        assert_eq!(
            var_with("PROOFYLOOPS_AUTO_BUILD", lookup(values)),
            Ok("0".to_string())
        );
    }

    #[test]
    fn legacy_value_is_used_only_when_canonical_is_absent() {
        let values = HashMap::from([("PROOFPATCH_AUTO_BUILD", Ok("0"))]);
        assert_eq!(
            var_with("PROOFYLOOPS_AUTO_BUILD", lookup(values)),
            Ok("0".to_string())
        );
    }

    #[test]
    fn empty_or_non_unicode_canonical_value_does_not_fall_back() {
        let empty = HashMap::from([
            ("PROOFYLOOPS_AUTO_BUILD", Ok("")),
            ("PROOFPATCH_AUTO_BUILD", Ok("1")),
        ]);
        assert_eq!(
            var_with("PROOFYLOOPS_AUTO_BUILD", lookup(empty)),
            Ok(String::new())
        );

        let non_unicode = HashMap::from([
            (
                "PROOFYLOOPS_AUTO_BUILD",
                Err(VarError::NotUnicode("x".into())),
            ),
            ("PROOFPATCH_AUTO_BUILD", Ok("1")),
        ]);
        assert!(matches!(
            var_with("PROOFYLOOPS_AUTO_BUILD", lookup(non_unicode)),
            Err(VarError::NotUnicode(_))
        ));
    }

    #[test]
    fn other_environment_names_are_not_rewritten() {
        let values = HashMap::from([("LAKE", Ok("lake"))]);
        assert_eq!(var_with("LAKE", lookup(values)), Ok("lake".to_string()));
    }
}
