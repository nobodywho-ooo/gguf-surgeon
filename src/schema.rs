use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::value::{GgufValue, GgufValueType};

#[derive(Debug, Clone, Deserialize)]
pub struct Schema {
    pub version: u32,
    #[serde(default)]
    pub applies_to: Vec<u32>,
    #[serde(default)]
    pub rules: BTreeMap<String, Rule>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Rule {
    #[serde(rename = "type")]
    pub type_spec: Option<String>,
    #[serde(rename = "enum")]
    pub allowed: Option<Vec<serde_json::Value>>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub severity: Option<Severity>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Error,
    Warning,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// Derived from the GGUF spec itself; failures produce an unloadable file.
    Format,
    /// Derived from the loaded schema overlay; overridable with `--force`.
    Schema,
}

#[derive(Debug, Clone)]
pub struct Violation {
    pub origin: Origin,
    pub key: String,
    pub severity: Severity,
    pub message: String,
}

impl Schema {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("could not read schema: {}", path.display()))?;
        Self::parse(&text)
    }

    pub fn parse(text: &str) -> Result<Self> {
        let schema: Schema = serde_json::from_str(text).context("invalid schema JSON")?;
        if schema.version != 1 {
            bail!("schema version {} not supported (only v1)", schema.version);
        }
        Ok(schema)
    }

    pub fn applies_to_version(&self, gguf_version: u32) -> bool {
        self.applies_to.is_empty() || self.applies_to.contains(&gguf_version)
    }

    pub fn validate(&self, metadata: &[(String, GgufValue)]) -> Vec<Violation> {
        let mut out = Vec::new();
        for (key, value) in metadata {
            if let Some(rule) = self.rules.get(key) {
                let sev = rule.severity.unwrap_or(Severity::Error);
                check_rule(key, value, rule, sev, &mut out);
            }
        }
        out
    }
}

fn check_rule(key: &str, value: &GgufValue, rule: &Rule, sev: Severity, out: &mut Vec<Violation>) {
    if let Some(spec) = &rule.type_spec
        && !type_spec_matches(spec, value)
    {
        out.push(Violation {
            origin: Origin::Schema,
            key: key.to_string(),
            severity: sev,
            message: format!("expected type {spec}, got {}", describe_type(value)),
        });
    }
    if let Some(allowed) = &rule.allowed
        && !allowed.iter().any(|j| json_eq_value(j, value))
    {
        let listed = allowed
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        out.push(Violation {
            origin: Origin::Schema,
            key: key.to_string(),
            severity: sev,
            message: format!("value not in allowed set [{listed}]"),
        });
    }
    if let Some(n) = numeric_value(value) {
        if let Some(min) = rule.min
            && n < min
        {
            out.push(Violation {
                origin: Origin::Schema,
                key: key.to_string(),
                severity: sev,
                message: format!("value {n} below min {min}"),
            });
        }
        if let Some(max) = rule.max
            && n > max
        {
            out.push(Violation {
                origin: Origin::Schema,
                key: key.to_string(),
                severity: sev,
                message: format!("value {n} above max {max}"),
            });
        }
    }
    if let Some(len) = length_of(value) {
        if let Some(min) = rule.min_length
            && len < min
        {
            out.push(Violation {
                origin: Origin::Schema,
                key: key.to_string(),
                severity: sev,
                message: format!("length {len} below min_length {min}"),
            });
        }
        if let Some(max) = rule.max_length
            && len > max
        {
            out.push(Violation {
                origin: Origin::Schema,
                key: key.to_string(),
                severity: sev,
                message: format!("length {len} above max_length {max}"),
            });
        }
    }
}

fn describe_type(v: &GgufValue) -> String {
    match v {
        GgufValue::Array(a) => format!("array<{}>", a.element_type.as_str()),
        _ => v.ty().as_str().to_string(),
    }
}

fn type_spec_matches(spec: &str, v: &GgufValue) -> bool {
    if let Some(inner) = spec.strip_prefix("array<").and_then(|s| s.strip_suffix('>')) {
        if let GgufValue::Array(a) = v {
            return GgufValueType::parse_name(inner.trim()) == Some(a.element_type);
        }
        return false;
    }
    GgufValueType::parse_name(spec) == Some(v.ty())
}

fn json_eq_value(j: &serde_json::Value, v: &GgufValue) -> bool {
    use serde_json::Value as J;
    match (j, v) {
        (J::String(s), GgufValue::String(t)) => s == t,
        (J::Bool(a), GgufValue::Bool(b)) => a == b,
        (J::Number(_), _) => match v {
            GgufValue::Uint8(x) => j.as_u64() == Some(u64::from(*x)),
            GgufValue::Int8(x) => j.as_i64() == Some(i64::from(*x)),
            GgufValue::Uint16(x) => j.as_u64() == Some(u64::from(*x)),
            GgufValue::Int16(x) => j.as_i64() == Some(i64::from(*x)),
            GgufValue::Uint32(x) => j.as_u64() == Some(u64::from(*x)),
            GgufValue::Int32(x) => j.as_i64() == Some(i64::from(*x)),
            GgufValue::Uint64(x) => j.as_u64() == Some(*x),
            GgufValue::Int64(x) => j.as_i64() == Some(*x),
            GgufValue::Float32(x) => j.as_f64() == Some(f64::from(*x)),
            GgufValue::Float64(x) => j.as_f64() == Some(*x),
            _ => false,
        },
        _ => false,
    }
}

fn numeric_value(v: &GgufValue) -> Option<f64> {
    Some(match v {
        GgufValue::Uint8(n) => f64::from(*n),
        GgufValue::Int8(n) => f64::from(*n),
        GgufValue::Uint16(n) => f64::from(*n),
        GgufValue::Int16(n) => f64::from(*n),
        GgufValue::Uint32(n) => f64::from(*n),
        GgufValue::Int32(n) => f64::from(*n),
        GgufValue::Uint64(n) => *n as f64,
        GgufValue::Int64(n) => *n as f64,
        GgufValue::Float32(n) => f64::from(*n),
        GgufValue::Float64(n) => *n,
        _ => return None,
    })
}

fn length_of(v: &GgufValue) -> Option<usize> {
    Some(match v {
        GgufValue::String(s) => s.chars().count(),
        GgufValue::Array(a) => a.elements.len(),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{GgufArray, GgufValueType};

    fn kv(k: &str, v: GgufValue) -> (String, GgufValue) {
        (k.to_string(), v)
    }

    #[test]
    fn parse_basic_schema() {
        let json = r#"{
            "version": 1,
            "applies_to": [3],
            "rules": {
                "general.architecture": {
                    "type": "string",
                    "enum": ["llama", "mistral"]
                }
            }
        }"#;
        let s = Schema::parse(json).unwrap();
        assert_eq!(s.version, 1);
        assert!(s.applies_to_version(3));
        assert!(!s.applies_to_version(99));
        assert_eq!(s.rules.len(), 1);
    }

    #[test]
    fn rejects_wrong_schema_version() {
        let json = r#"{ "version": 99 }"#;
        assert!(Schema::parse(json).is_err());
    }

    #[test]
    fn enum_violation_marked_error_by_default() {
        let s = Schema::parse(
            r#"{ "version": 1, "rules": { "arch": { "enum": ["llama"] } } }"#,
        )
        .unwrap();
        let v = s.validate(&[kv("arch", GgufValue::String("mistral".to_string()))]);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].severity, Severity::Error);
        assert!(v[0].message.contains("not in allowed"));
    }

    #[test]
    fn enum_match_passes() {
        let s = Schema::parse(
            r#"{ "version": 1, "rules": { "arch": { "enum": ["llama", "mistral"] } } }"#,
        )
        .unwrap();
        let v = s.validate(&[kv("arch", GgufValue::String("llama".to_string()))]);
        assert!(v.is_empty());
    }

    #[test]
    fn min_max_violations() {
        let s = Schema::parse(
            r#"{ "version": 1, "rules": { "ctx": { "min": 1, "max": 100, "severity": "warning" } } }"#,
        )
        .unwrap();
        let v = s.validate(&[kv("ctx", GgufValue::Uint32(200))]);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].severity, Severity::Warning);
        assert!(v[0].message.contains("above max"));
    }

    #[test]
    fn length_constraints() {
        let s = Schema::parse(
            r#"{ "version": 1, "rules": { "vocab": { "min_length": 1 } } }"#,
        )
        .unwrap();
        let empty = GgufArray {
            element_type: GgufValueType::String,
            elements: vec![],
        };
        let v = s.validate(&[kv("vocab", GgufValue::Array(empty))]);
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("below min_length"));
    }

    #[test]
    fn type_mismatch_detected() {
        let s = Schema::parse(
            r#"{ "version": 1, "rules": { "arch": { "type": "string" } } }"#,
        )
        .unwrap();
        let v = s.validate(&[kv("arch", GgufValue::Uint32(0))]);
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("expected type string"));
    }

    #[test]
    fn unknown_keys_are_ignored() {
        let s = Schema::parse(r#"{ "version": 1, "rules": {} }"#).unwrap();
        let v = s.validate(&[kv("anything", GgufValue::Uint32(0))]);
        assert!(v.is_empty());
    }
}
