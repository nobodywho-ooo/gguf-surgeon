use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::format::GgufFile;
use crate::value::{GgufArray, GgufValue, GgufValueType};

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum Op {
    Set {
        key: String,
        value: serde_json::Value,
    },
    Add {
        key: String,
        #[serde(rename = "type")]
        type_spec: String,
        value: serde_json::Value,
    },
    Rm {
        key: String,
    },
}

pub type Patch = Vec<Op>;

pub fn parse_patch(json: &str) -> Result<Patch> {
    serde_json::from_str(json).context("could not parse patch as JSON array of operations")
}

pub fn apply(file: &mut GgufFile, patch: &Patch) -> Result<()> {
    for (i, op) in patch.iter().enumerate() {
        apply_one(file, op).with_context(|| format!("patch op #{i} failed"))?;
    }
    Ok(())
}

fn apply_one(file: &mut GgufFile, op: &Op) -> Result<()> {
    match op {
        Op::Set { key, value } => {
            let pos = file
                .metadata
                .iter()
                .position(|(k, _)| k == key)
                .with_context(|| format!("set: key not found: {key}"))?;
            let target = type_spec_of(&file.metadata[pos].1);
            let new_value = json_to_gguf(value, &target)?;
            file.metadata[pos].1 = new_value;
        }
        Op::Add {
            key,
            type_spec,
            value,
        } => {
            if file.metadata.iter().any(|(k, _)| k == key) {
                bail!("add: key already exists: {key}");
            }
            let target = parse_type_spec(type_spec)
                .with_context(|| format!("add: unknown type spec: {type_spec}"))?;
            let new_value = json_to_gguf(value, &target)?;
            file.metadata.push((key.clone(), new_value));
        }
        Op::Rm { key } => {
            let pos = file
                .metadata
                .iter()
                .position(|(k, _)| k == key)
                .with_context(|| format!("rm: key not found: {key}"))?;
            file.metadata.remove(pos);
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum TypeSpec {
    Scalar(GgufValueType),
    Array(GgufValueType),
}

fn type_spec_of(value: &GgufValue) -> TypeSpec {
    match value {
        GgufValue::Array(a) => TypeSpec::Array(a.element_type),
        _ => TypeSpec::Scalar(value.ty()),
    }
}

fn parse_type_spec(s: &str) -> Option<TypeSpec> {
    if let Some(inner) = s.strip_prefix("array<").and_then(|s| s.strip_suffix('>')) {
        let elem = GgufValueType::parse_name(inner.trim())?;
        if matches!(elem, GgufValueType::Array) {
            return None;
        }
        Some(TypeSpec::Array(elem))
    } else {
        let ty = GgufValueType::parse_name(s)?;
        if matches!(ty, GgufValueType::Array) {
            return None;
        }
        Some(TypeSpec::Scalar(ty))
    }
}

fn json_to_gguf(value: &serde_json::Value, target: &TypeSpec) -> Result<GgufValue> {
    match target {
        TypeSpec::Scalar(ty) => json_to_scalar(value, *ty),
        TypeSpec::Array(elem_ty) => {
            let arr = value
                .as_array()
                .with_context(|| format!("expected JSON array for array<{}>", elem_ty.as_str()))?;
            let elements = arr
                .iter()
                .map(|v| json_to_scalar(v, *elem_ty))
                .collect::<Result<Vec<_>>>()?;
            Ok(GgufValue::Array(GgufArray {
                element_type: *elem_ty,
                elements,
            }))
        }
    }
}

fn json_to_scalar(value: &serde_json::Value, ty: GgufValueType) -> Result<GgufValue> {
    use serde_json::Value as J;
    Ok(match ty {
        GgufValueType::Uint8 => GgufValue::Uint8(int_in_range::<u8>(value)?),
        GgufValueType::Int8 => GgufValue::Int8(int_in_range::<i8>(value)?),
        GgufValueType::Uint16 => GgufValue::Uint16(int_in_range::<u16>(value)?),
        GgufValueType::Int16 => GgufValue::Int16(int_in_range::<i16>(value)?),
        GgufValueType::Uint32 => GgufValue::Uint32(int_in_range::<u32>(value)?),
        GgufValueType::Int32 => GgufValue::Int32(int_in_range::<i32>(value)?),
        GgufValueType::Uint64 => GgufValue::Uint64(value.as_u64().context("expected unsigned integer")?),
        GgufValueType::Int64 => GgufValue::Int64(value.as_i64().context("expected signed integer")?),
        GgufValueType::Float32 => {
            let f = value.as_f64().context("expected number")?;
            GgufValue::Float32(f as f32)
        }
        GgufValueType::Float64 => GgufValue::Float64(value.as_f64().context("expected number")?),
        GgufValueType::Bool => GgufValue::Bool(value.as_bool().context("expected boolean")?),
        GgufValueType::String => match value {
            J::String(s) => GgufValue::String(s.clone()),
            _ => bail!("expected JSON string"),
        },
        GgufValueType::Array => bail!("nested arrays are not allowed"),
    })
}

fn int_in_range<T>(value: &serde_json::Value) -> Result<T>
where
    T: TryFrom<i64> + TryFrom<u64>,
{
    if let Some(n) = value.as_u64() {
        T::try_from(n).map_err(|_| anyhow::anyhow!("integer {n} out of range for target type"))
    } else if let Some(n) = value.as_i64() {
        T::try_from(n).map_err(|_| anyhow::anyhow!("integer {n} out of range for target type"))
    } else {
        bail!("expected integer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_patch() {
        let json = r#"[
            {"op": "set", "key": "a", "value": 42},
            {"op": "add", "key": "b", "type": "bool", "value": true},
            {"op": "rm",  "key": "c"}
        ]"#;
        let patch = parse_patch(json).unwrap();
        assert_eq!(patch.len(), 3);
    }

    #[test]
    fn parse_type_spec_handles_arrays() {
        match parse_type_spec("array<u32>").unwrap() {
            TypeSpec::Array(GgufValueType::Uint32) => {}
            _ => panic!(),
        }
        match parse_type_spec("array<string>").unwrap() {
            TypeSpec::Array(GgufValueType::String) => {}
            _ => panic!(),
        }
        assert!(parse_type_spec("array<array<u32>>").is_none());
        assert!(parse_type_spec("array<bogus>").is_none());
        assert!(parse_type_spec("array").is_none()); // bare array needs element type
    }

    #[test]
    fn applies_set_add_rm_in_order() {
        let mut f = GgufFile {
            version: 3,
            tensor_count: 0,
            metadata: vec![
                ("a".to_string(), GgufValue::Uint32(1)),
                ("b".to_string(), GgufValue::String("old".to_string())),
            ],
            tensors: vec![],
            alignment: 32,
            header_end: 0,
            tensor_data_offset: 0,
        };
        let patch = parse_patch(
            r#"[
            {"op": "set", "key": "a", "value": 99},
            {"op": "add", "key": "shape", "type": "array<u64>", "value": [4, 8, 16]},
            {"op": "rm",  "key": "b"}
        ]"#,
        )
        .unwrap();
        apply(&mut f, &patch).unwrap();

        assert_eq!(f.metadata.len(), 2);
        assert_eq!(
            f.metadata.iter().find(|(k, _)| k == "a").unwrap().1,
            GgufValue::Uint32(99)
        );
        let shape = &f.metadata.iter().find(|(k, _)| k == "shape").unwrap().1;
        let GgufValue::Array(arr) = shape else {
            panic!("expected array");
        };
        assert_eq!(arr.element_type, GgufValueType::Uint64);
        assert_eq!(arr.elements.len(), 3);
        assert_eq!(arr.elements[2], GgufValue::Uint64(16));
    }

    #[test]
    fn rejects_out_of_range_integer() {
        let mut f = GgufFile {
            version: 3,
            tensor_count: 0,
            metadata: vec![("a".to_string(), GgufValue::Uint8(0))],
            tensors: vec![],
            alignment: 32,
            header_end: 0,
            tensor_data_offset: 0,
        };
        let patch = parse_patch(r#"[{"op": "set", "key": "a", "value": 99999}]"#).unwrap();
        assert!(apply(&mut f, &patch).is_err());
    }

    #[test]
    fn rejects_set_on_missing_key() {
        let mut f = GgufFile {
            version: 3,
            tensor_count: 0,
            metadata: vec![],
            tensors: vec![],
            alignment: 32,
            header_end: 0,
            tensor_data_offset: 0,
        };
        let patch = parse_patch(r#"[{"op": "set", "key": "x", "value": 1}]"#).unwrap();
        assert!(apply(&mut f, &patch).is_err());
    }
}
