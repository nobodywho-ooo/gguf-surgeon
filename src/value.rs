#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    pub fn from_u32(tag: u32) -> Option<Self> {
        Some(match tag {
            0 => Self::Uint8,
            1 => Self::Int8,
            2 => Self::Uint16,
            3 => Self::Int16,
            4 => Self::Uint32,
            5 => Self::Int32,
            6 => Self::Float32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::Uint64,
            11 => Self::Int64,
            12 => Self::Float64,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Uint8 => "u8",
            Self::Int8 => "i8",
            Self::Uint16 => "u16",
            Self::Int16 => "i16",
            Self::Uint32 => "u32",
            Self::Int32 => "i32",
            Self::Float32 => "f32",
            Self::Bool => "bool",
            Self::String => "string",
            Self::Array => "array",
            Self::Uint64 => "u64",
            Self::Int64 => "i64",
            Self::Float64 => "f64",
        }
    }

    pub fn parse_name(name: &str) -> Option<Self> {
        Some(match name {
            "u8" | "uint8" => Self::Uint8,
            "i8" | "int8" => Self::Int8,
            "u16" | "uint16" => Self::Uint16,
            "i16" | "int16" => Self::Int16,
            "u32" | "uint32" => Self::Uint32,
            "i32" | "int32" => Self::Int32,
            "u64" | "uint64" => Self::Uint64,
            "i64" | "int64" => Self::Int64,
            "f32" | "float32" | "float" => Self::Float32,
            "f64" | "float64" | "double" => Self::Float64,
            "bool" => Self::Bool,
            "string" | "str" => Self::String,
            "array" => Self::Array,
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_name_round_trips_canonical() {
        for ty in [
            GgufValueType::Uint8,
            GgufValueType::Int8,
            GgufValueType::Uint16,
            GgufValueType::Int16,
            GgufValueType::Uint32,
            GgufValueType::Int32,
            GgufValueType::Uint64,
            GgufValueType::Int64,
            GgufValueType::Float32,
            GgufValueType::Float64,
            GgufValueType::Bool,
            GgufValueType::String,
            GgufValueType::Array,
        ] {
            assert_eq!(GgufValueType::parse_name(ty.as_str()), Some(ty));
        }
    }

    #[test]
    fn parse_name_accepts_aliases() {
        assert_eq!(GgufValueType::parse_name("uint32"), Some(GgufValueType::Uint32));
        assert_eq!(GgufValueType::parse_name("float"), Some(GgufValueType::Float32));
        assert_eq!(GgufValueType::parse_name("double"), Some(GgufValueType::Float64));
        assert_eq!(GgufValueType::parse_name("str"), Some(GgufValueType::String));
    }

    #[test]
    fn parse_name_rejects_unknown() {
        assert_eq!(GgufValueType::parse_name("nope"), None);
        assert_eq!(GgufValueType::parse_name(""), None);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(GgufArray),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufValue {
    pub fn ty(&self) -> GgufValueType {
        match self {
            Self::Uint8(_) => GgufValueType::Uint8,
            Self::Int8(_) => GgufValueType::Int8,
            Self::Uint16(_) => GgufValueType::Uint16,
            Self::Int16(_) => GgufValueType::Int16,
            Self::Uint32(_) => GgufValueType::Uint32,
            Self::Int32(_) => GgufValueType::Int32,
            Self::Float32(_) => GgufValueType::Float32,
            Self::Bool(_) => GgufValueType::Bool,
            Self::String(_) => GgufValueType::String,
            Self::Array(_) => GgufValueType::Array,
            Self::Uint64(_) => GgufValueType::Uint64,
            Self::Int64(_) => GgufValueType::Int64,
            Self::Float64(_) => GgufValueType::Float64,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        if let Self::Uint32(n) = *self { Some(n) } else { None }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgufArray {
    pub element_type: GgufValueType,
    pub elements: Vec<GgufValue>,
}
