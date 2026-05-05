use std::collections::HashSet;

use crate::value::GgufValue;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Diff {
    pub additions: Vec<(String, GgufValue)>,
    pub removals: Vec<(String, GgufValue)>,
    pub changes: Vec<(String, GgufValue, GgufValue)>,
}

impl Diff {
    pub fn between(before: &[(String, GgufValue)], after: &[(String, GgufValue)]) -> Self {
        let before_keys: HashSet<&str> = before.iter().map(|(k, _)| k.as_str()).collect();
        let after_keys: HashSet<&str> = after.iter().map(|(k, _)| k.as_str()).collect();

        let mut diff = Diff::default();

        for (k, v) in after {
            if !before_keys.contains(k.as_str()) {
                diff.additions.push((k.clone(), v.clone()));
            }
        }
        for (k, v) in before {
            if !after_keys.contains(k.as_str()) {
                diff.removals.push((k.clone(), v.clone()));
            }
        }
        for (k, old) in before {
            if let Some((_, new)) = after.iter().find(|(k2, _)| k2 == k)
                && old != new
            {
                diff.changes.push((k.clone(), old.clone(), new.clone()));
            }
        }

        diff
    }

    pub fn is_empty(&self) -> bool {
        self.additions.is_empty() && self.removals.is_empty() && self.changes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kv(k: &str, v: GgufValue) -> (String, GgufValue) {
        (k.to_string(), v)
    }

    #[test]
    fn empty_when_identical() {
        let m = vec![kv("a", GgufValue::Uint32(1))];
        let d = Diff::between(&m, &m);
        assert!(d.is_empty());
    }

    #[test]
    fn detects_addition() {
        let before = vec![kv("a", GgufValue::Uint32(1))];
        let after = vec![
            kv("a", GgufValue::Uint32(1)),
            kv("b", GgufValue::Bool(true)),
        ];
        let d = Diff::between(&before, &after);
        assert_eq!(d.additions, vec![kv("b", GgufValue::Bool(true))]);
        assert!(d.removals.is_empty());
        assert!(d.changes.is_empty());
    }

    #[test]
    fn detects_removal() {
        let before = vec![
            kv("a", GgufValue::Uint32(1)),
            kv("b", GgufValue::Bool(true)),
        ];
        let after = vec![kv("a", GgufValue::Uint32(1))];
        let d = Diff::between(&before, &after);
        assert_eq!(d.removals, vec![kv("b", GgufValue::Bool(true))]);
        assert!(d.additions.is_empty());
        assert!(d.changes.is_empty());
    }

    #[test]
    fn detects_change() {
        let before = vec![kv("a", GgufValue::Uint32(1))];
        let after = vec![kv("a", GgufValue::Uint32(2))];
        let d = Diff::between(&before, &after);
        assert_eq!(
            d.changes,
            vec![(
                "a".to_string(),
                GgufValue::Uint32(1),
                GgufValue::Uint32(2),
            )]
        );
        assert!(d.additions.is_empty());
        assert!(d.removals.is_empty());
    }
}
