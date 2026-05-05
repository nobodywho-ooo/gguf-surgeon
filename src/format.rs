use std::collections::HashSet;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::error::Error;
use crate::schema::{Origin, Severity, Violation};
use crate::value::{GgufArray, GgufValue, GgufValueType};
use crate::version;

const MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;
const ALIGNMENT_KEY: &str = "general.alignment";
const PADDING_KEY: &str = "general.padding";

/// Default slack step for the sentinel padding key. The header is grown to a multiple of
/// this value so subsequent edits within the budget can use the header-overwrite save path.
pub const DEFAULT_PADDING_STEP: u64 = 64 * 1024;

/// Encoded byte overhead of the sentinel padding key itself (key length prefix + key bytes
/// + value type tag + array element type + array length prefix). The actual byte count
/// of zeros is appended on top of this.
const PADDING_OVERHEAD: u64 = 8 + 15 + 4 + 4 + 8;

#[derive(Debug, Clone)]
pub struct GgufFile {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata: Vec<(String, GgufValue)>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: u64,
    /// Byte offset right after the tensor_info table (where alignment padding starts).
    pub header_end: u64,
    /// Absolute byte offset where tensor data begins (after alignment padding).
    pub tensor_data_offset: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
}

fn align_up(n: u64, align: u64) -> u64 {
    if align == 0 {
        return n;
    }
    n.div_ceil(align) * align
}

fn alignment_from_metadata(metadata: &[(String, GgufValue)]) -> u64 {
    metadata
        .iter()
        .find(|(k, _)| k == ALIGNMENT_KEY)
        .and_then(|(_, v)| v.as_u32())
        .map(u64::from)
        .unwrap_or(DEFAULT_ALIGNMENT)
}

impl GgufFile {
    pub fn read(path: &Path) -> Result<Self, Error> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Self::parse(&mmap)
    }

    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(data);

        let magic_bytes: [u8; 4] = r.take(4)?.try_into().unwrap();
        if &magic_bytes != MAGIC {
            return Err(Error::BadMagic(magic_bytes));
        }

        let version = r.u32()?;
        if !version::is_supported(version) {
            return Err(Error::UnsupportedVersion {
                found: version,
                supported: version::SUPPORTED_VERSIONS,
            });
        }

        let tensor_count = r.u64()?;
        let kv_count = r.u64()?;

        if kv_count > r.remaining() as u64 {
            return Err(Error::OversizedDeclaration {
                declared: kv_count,
                remaining: r.remaining() as u64,
            });
        }

        let mut metadata = Vec::with_capacity(kv_count as usize);
        for _ in 0..kv_count {
            let key = r.string()?;
            let ty_tag = r.u32()?;
            let ty = GgufValueType::from_u32(ty_tag).ok_or(Error::UnknownValueType(ty_tag))?;
            let value = r.value(ty)?;
            metadata.push((key, value));
        }

        if tensor_count > r.remaining() as u64 {
            return Err(Error::OversizedDeclaration {
                declared: tensor_count,
                remaining: r.remaining() as u64,
            });
        }
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            tensors.push(r.tensor_info()?);
        }

        let alignment = alignment_from_metadata(&metadata);

        let header_end = r.pos() as u64;
        let tensor_data_offset = align_up(header_end, alignment);

        Ok(GgufFile {
            version,
            tensor_count,
            metadata,
            tensors,
            alignment,
            header_end,
            tensor_data_offset,
        })
    }

    /// Insert (or replace) the `general.padding` sentinel key so the encoded header rounds
    /// up to a multiple of `step` bytes. This reserves slack for future small edits, which
    /// can then be saved via the header-overwrite path without copying tensor data.
    /// Pass `step = 0` to remove any existing padding key.
    pub fn ensure_padding(&mut self, step: u64) {
        self.metadata.retain(|(k, _)| k != PADDING_KEY);
        if step == 0 {
            return;
        }
        let raw = self.encoded_size_unaligned();
        let min_total_with_padding = raw + PADDING_OVERHEAD;
        let target = min_total_with_padding.div_ceil(step) * step;
        let zeros = (target - min_total_with_padding) as usize;
        let elements = std::iter::repeat_n(GgufValue::Uint8(0), zeros).collect();
        self.metadata.push((
            PADDING_KEY.to_string(),
            GgufValue::Array(GgufArray {
                element_type: GgufValueType::Uint8,
                elements,
            }),
        ));
    }

    /// Size in bytes of the metadata block + tensor_info table, before alignment padding.
    pub fn encoded_size_unaligned(&self) -> u64 {
        let mut size: u64 = 4 + 4 + 8 + 8;
        for (k, v) in &self.metadata {
            size += 8 + k.len() as u64 + 4 + value_encoded_size(v);
        }
        for t in &self.tensors {
            size += 8 + t.name.len() as u64 + 4 + 8 * t.dims.len() as u64 + 4 + 8;
        }
        size
    }

    /// Run format-level validation: spec invariants that must hold for the file to load.
    /// All violations are returned with `Origin::Format` and `Severity::Error`; format errors
    /// are unconditional (no `--force` override) because the file would be unloadable.
    pub fn validate_format(&self) -> Vec<Violation> {
        let mut out = Vec::new();
        let mut seen: HashSet<&str> = HashSet::new();
        for (key, _) in &self.metadata {
            if !seen.insert(key.as_str()) {
                out.push(Violation {
                    origin: Origin::Format,
                    key: key.clone(),
                    severity: Severity::Error,
                    message: "duplicate metadata key".to_string(),
                });
            }
        }
        out
    }

    /// Encode the header region (magic through minimum alignment padding) into a byte vector.
    /// Counts and alignment padding are derived from the current state, not from stale fields.
    /// Reserving extra slack for future in-place edits is a save-path concern (sentinel key
    /// or alignment adjustment) and is intentionally not handled here.
    pub fn encode_header(&self) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut w = Writer::new(&mut out);
            w.bytes(MAGIC);
            w.u32(self.version);
            w.u64(self.tensors.len() as u64);
            w.u64(self.metadata.len() as u64);
            for (key, value) in &self.metadata {
                w.string(key);
                w.u32(value.ty() as u32);
                w.value(value);
            }
            for t in &self.tensors {
                w.string(&t.name);
                let n_dims: u32 = t.dims.len().try_into().expect("dim count fits in u32");
                w.u32(n_dims);
                for d in &t.dims {
                    w.u64(*d);
                }
                w.u32(t.ggml_type);
                w.u64(t.offset);
            }
        }
        let alignment = alignment_from_metadata(&self.metadata);
        let target = align_up(out.len() as u64, alignment);
        out.resize(target as usize, 0);
        out
    }
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        if self.remaining() < n {
            return Err(Error::Truncated {
                offset: self.pos as u64,
            });
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.take(1)?[0])
    }
    fn i8(&mut self) -> Result<i8, Error> {
        Ok(self.take(1)?[0] as i8)
    }
    fn u16(&mut self) -> Result<u16, Error> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn i16(&mut self) -> Result<i16, Error> {
        Ok(i16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32, Error> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Result<i64, Error> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, Error> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64, Error> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn bool_(&mut self) -> Result<bool, Error> {
        Ok(self.u8()? != 0)
    }

    fn string(&mut self) -> Result<String, Error> {
        let len = self.u64()?;
        if len > self.remaining() as u64 {
            return Err(Error::OversizedDeclaration {
                declared: len,
                remaining: self.remaining() as u64,
            });
        }
        let start = self.pos as u64;
        let bytes = self.take(len as usize)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| Error::InvalidUtf8 { offset: start })
    }

    fn value(&mut self, ty: GgufValueType) -> Result<GgufValue, Error> {
        Ok(match ty {
            GgufValueType::Uint8 => GgufValue::Uint8(self.u8()?),
            GgufValueType::Int8 => GgufValue::Int8(self.i8()?),
            GgufValueType::Uint16 => GgufValue::Uint16(self.u16()?),
            GgufValueType::Int16 => GgufValue::Int16(self.i16()?),
            GgufValueType::Uint32 => GgufValue::Uint32(self.u32()?),
            GgufValueType::Int32 => GgufValue::Int32(self.i32()?),
            GgufValueType::Float32 => GgufValue::Float32(self.f32()?),
            GgufValueType::Bool => GgufValue::Bool(self.bool_()?),
            GgufValueType::String => GgufValue::String(self.string()?),
            GgufValueType::Array => GgufValue::Array(self.array()?),
            GgufValueType::Uint64 => GgufValue::Uint64(self.u64()?),
            GgufValueType::Int64 => GgufValue::Int64(self.i64()?),
            GgufValueType::Float64 => GgufValue::Float64(self.f64()?),
        })
    }

    fn array(&mut self) -> Result<GgufArray, Error> {
        let elem_tag = self.u32()?;
        let element_type =
            GgufValueType::from_u32(elem_tag).ok_or(Error::UnknownValueType(elem_tag))?;
        let len = self.u64()?;
        if len > self.remaining() as u64 {
            return Err(Error::OversizedDeclaration {
                declared: len,
                remaining: self.remaining() as u64,
            });
        }
        let mut elements = Vec::with_capacity(len as usize);
        for _ in 0..len {
            elements.push(self.value(element_type)?);
        }
        Ok(GgufArray {
            element_type,
            elements,
        })
    }

    fn tensor_info(&mut self) -> Result<TensorInfo, Error> {
        let name = self.string()?;
        let n_dims = self.u32()? as u64;
        if n_dims.saturating_mul(8) > self.remaining() as u64 {
            return Err(Error::OversizedDeclaration {
                declared: n_dims,
                remaining: self.remaining() as u64,
            });
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(self.u64()?);
        }
        let ggml_type = self.u32()?;
        let offset = self.u64()?;
        Ok(TensorInfo {
            name,
            dims,
            ggml_type,
            offset,
        })
    }
}

fn value_encoded_size(v: &GgufValue) -> u64 {
    match v {
        GgufValue::Uint8(_) | GgufValue::Int8(_) | GgufValue::Bool(_) => 1,
        GgufValue::Uint16(_) | GgufValue::Int16(_) => 2,
        GgufValue::Uint32(_) | GgufValue::Int32(_) | GgufValue::Float32(_) => 4,
        GgufValue::Uint64(_) | GgufValue::Int64(_) | GgufValue::Float64(_) => 8,
        GgufValue::String(s) => 8 + s.len() as u64,
        GgufValue::Array(a) => {
            4 + 8 + a.elements.iter().map(value_encoded_size).sum::<u64>()
        }
    }
}

struct Writer<'a> {
    out: &'a mut Vec<u8>,
}

impl<'a> Writer<'a> {
    fn new(out: &'a mut Vec<u8>) -> Self {
        Self { out }
    }

    fn bytes(&mut self, b: &[u8]) {
        self.out.extend_from_slice(b);
    }
    fn u8(&mut self, v: u8) {
        self.out.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }
    fn i8_(&mut self, v: i8) {
        self.out.push(v as u8);
    }
    fn i16(&mut self, v: i16) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }
    fn bool_(&mut self, v: bool) {
        self.out.push(u8::from(v));
    }

    fn string(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.bytes(s.as_bytes());
    }

    fn value(&mut self, v: &GgufValue) {
        match v {
            GgufValue::Uint8(n) => self.u8(*n),
            GgufValue::Int8(n) => self.i8_(*n),
            GgufValue::Uint16(n) => self.u16(*n),
            GgufValue::Int16(n) => self.i16(*n),
            GgufValue::Uint32(n) => self.u32(*n),
            GgufValue::Int32(n) => self.i32(*n),
            GgufValue::Float32(n) => self.f32(*n),
            GgufValue::Bool(b) => self.bool_(*b),
            GgufValue::String(s) => self.string(s),
            GgufValue::Array(a) => {
                self.u32(a.element_type as u32);
                self.u64(a.elements.len() as u64);
                for e in &a.elements {
                    self.value(e);
                }
            }
            GgufValue::Uint64(n) => self.u64(*n),
            GgufValue::Int64(n) => self.i64(*n),
            GgufValue::Float64(n) => self.f64(*n),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_str(b: &mut Vec<u8>, s: &[u8]) {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s);
    }

    fn put_kv_header(b: &mut Vec<u8>, key: &[u8], ty: GgufValueType) {
        put_str(b, key);
        b.extend_from_slice(&(ty as u32).to_le_bytes());
    }

    fn header(version: u32, tensor_count: u64, kv_count: u64) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&version.to_le_bytes());
        b.extend_from_slice(&tensor_count.to_le_bytes());
        b.extend_from_slice(&kv_count.to_le_bytes());
        b
    }

    #[test]
    fn parses_minimal_file_with_mixed_types() {
        let mut b = header(3, 0, 4);

        put_kv_header(&mut b, b"answer", GgufValueType::Uint32);
        b.extend_from_slice(&42u32.to_le_bytes());

        put_kv_header(&mut b, b"general.architecture", GgufValueType::String);
        put_str(&mut b, b"llama");

        put_kv_header(&mut b, b"ratio", GgufValueType::Float32);
        b.extend_from_slice(&1.5f32.to_le_bytes());

        put_kv_header(&mut b, b"tokens", GgufValueType::Array);
        b.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        b.extend_from_slice(&3u64.to_le_bytes());
        put_str(&mut b, b"a");
        put_str(&mut b, b"bb");
        put_str(&mut b, b"ccc");

        let f = GgufFile::parse(&b).expect("parse");
        assert_eq!(f.version, 3);
        assert_eq!(f.tensor_count, 0);
        assert_eq!(f.metadata.len(), 4);

        assert_eq!(f.metadata[0].0, "answer");
        assert_eq!(f.metadata[0].1, GgufValue::Uint32(42));

        assert_eq!(f.metadata[1].0, "general.architecture");
        assert_eq!(f.metadata[1].1, GgufValue::String("llama".to_string()));

        assert_eq!(f.metadata[2].0, "ratio");
        assert_eq!(f.metadata[2].1, GgufValue::Float32(1.5));

        let GgufValue::Array(arr) = &f.metadata[3].1 else {
            panic!("expected Array");
        };
        assert_eq!(arr.element_type, GgufValueType::String);
        assert_eq!(arr.elements.len(), 3);
        assert_eq!(arr.elements[2], GgufValue::String("ccc".to_string()));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut b = vec![b'F', b'A', b'K', b'E'];
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        let err = GgufFile::parse(&b).unwrap_err();
        assert!(matches!(err, Error::BadMagic([b'F', b'A', b'K', b'E'])));
    }

    #[test]
    fn rejects_unsupported_version() {
        let b = header(99, 0, 0);
        let err = GgufFile::parse(&b).unwrap_err();
        assert!(matches!(err, Error::UnsupportedVersion { found: 99, .. }));
    }

    #[test]
    fn rejects_unknown_value_type_tag() {
        let mut b = header(3, 0, 1);
        put_str(&mut b, b"bad");
        b.extend_from_slice(&999u32.to_le_bytes()); // not a valid tag
        let err = GgufFile::parse(&b).unwrap_err();
        assert!(matches!(err, Error::UnknownValueType(999)));
    }

    #[test]
    fn rejects_oversized_array_length() {
        let mut b = header(3, 0, 1);
        put_kv_header(&mut b, b"big", GgufValueType::Array);
        b.extend_from_slice(&(GgufValueType::Uint8 as u32).to_le_bytes());
        b.extend_from_slice(&u64::MAX.to_le_bytes()); // way too many elements
        let err = GgufFile::parse(&b).unwrap_err();
        assert!(matches!(err, Error::OversizedDeclaration { .. }));
    }

    #[test]
    fn rejects_truncated_value() {
        let mut b = header(3, 0, 1);
        put_kv_header(&mut b, b"v", GgufValueType::Uint64);
        b.extend_from_slice(&[0u8, 0, 0]); // only 3 bytes, need 8
        let err = GgufFile::parse(&b).unwrap_err();
        assert!(matches!(err, Error::Truncated { .. }));
    }

    #[test]
    fn parses_tensor_info_and_alignment() {
        let mut b = header(3, 2, 1);

        // metadata: general.alignment = 64 (overrides default of 32)
        put_kv_header(&mut b, b"general.alignment", GgufValueType::Uint32);
        b.extend_from_slice(&64u32.to_le_bytes());

        // tensor 1: name="weights", dims=[4, 8], type=0, offset=0
        put_str(&mut b, b"weights");
        b.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        b.extend_from_slice(&4u64.to_le_bytes());
        b.extend_from_slice(&8u64.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // ggml_type
        b.extend_from_slice(&0u64.to_le_bytes()); // offset

        // tensor 2: name="bias", dims=[8], type=0, offset=128
        put_str(&mut b, b"bias");
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&8u64.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&128u64.to_le_bytes());

        let pre_padding_len = b.len() as u64;

        let f = GgufFile::parse(&b).expect("parse");
        assert_eq!(f.alignment, 64);
        assert_eq!(f.tensors.len(), 2);
        assert_eq!(f.tensors[0].name, "weights");
        assert_eq!(f.tensors[0].dims, vec![4, 8]);
        assert_eq!(f.tensors[1].name, "bias");
        assert_eq!(f.tensors[1].offset, 128);
        // tensor_data_offset should be the next multiple of 64 at or after the header end
        assert_eq!(f.tensor_data_offset, pre_padding_len.div_ceil(64) * 64);
        assert!(f.tensor_data_offset >= pre_padding_len);
    }

    #[test]
    fn defaults_alignment_to_32_when_unspecified() {
        let b = header(3, 0, 0);
        let f = GgufFile::parse(&b).expect("parse");
        assert_eq!(f.alignment, 32);
        assert_eq!(f.tensor_data_offset, 32); // header is 24 bytes, next multiple of 32 is 32
    }

    fn build_full_header_bytes() -> Vec<u8> {
        let mut b = header(3, 1, 4);

        put_kv_header(&mut b, b"general.architecture", GgufValueType::String);
        put_str(&mut b, b"llama");

        put_kv_header(&mut b, b"answer", GgufValueType::Uint32);
        b.extend_from_slice(&42u32.to_le_bytes());

        put_kv_header(&mut b, b"flag", GgufValueType::Bool);
        b.push(1);

        put_kv_header(&mut b, b"shape", GgufValueType::Array);
        b.extend_from_slice(&(GgufValueType::Uint64 as u32).to_le_bytes());
        b.extend_from_slice(&3u64.to_le_bytes());
        b.extend_from_slice(&7u64.to_le_bytes());
        b.extend_from_slice(&13u64.to_le_bytes());
        b.extend_from_slice(&21u64.to_le_bytes());

        // tensor_info: name="w", dims=[2,3], ggml_type=0, offset=0
        put_str(&mut b, b"w");
        b.extend_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        b.extend_from_slice(&3u64.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());

        // pad to default alignment 32
        let pad = (32 - b.len() % 32) % 32;
        b.extend(std::iter::repeat_n(0u8, pad));
        b
    }

    #[test]
    fn encode_round_trip_byte_identical() {
        let original = build_full_header_bytes();
        let f = GgufFile::parse(&original).expect("parse");
        let encoded = f.encode_header();
        assert_eq!(encoded.len(), original.len());
        assert_eq!(encoded, original);
    }

    #[test]
    fn encode_then_parse_preserves_content() {
        let original = build_full_header_bytes();
        let f1 = GgufFile::parse(&original).expect("parse 1");
        let encoded = f1.encode_header();
        let f2 = GgufFile::parse(&encoded).expect("parse 2");
        assert_eq!(f1.version, f2.version);
        assert_eq!(f1.metadata, f2.metadata);
        assert_eq!(f1.tensors, f2.tensors);
        assert_eq!(f1.alignment, f2.alignment);
        assert_eq!(f1.tensor_data_offset, f2.tensor_data_offset);
    }

    #[test]
    fn validate_format_detects_duplicate_keys() {
        let f = GgufFile {
            version: 3,
            tensor_count: 0,
            metadata: vec![
                ("a".to_string(), GgufValue::Uint32(1)),
                ("b".to_string(), GgufValue::Bool(true)),
                ("a".to_string(), GgufValue::Uint32(2)),
            ],
            tensors: vec![],
            alignment: 32,
            header_end: 0,
            tensor_data_offset: 0,
        };
        let v = f.validate_format();
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("duplicate"));
        assert_eq!(v[0].key, "a");
        assert_eq!(v[0].origin, Origin::Format);
    }

    #[test]
    fn validate_format_passes_unique_keys() {
        let f = GgufFile {
            version: 3,
            tensor_count: 0,
            metadata: vec![
                ("a".to_string(), GgufValue::Uint32(1)),
                ("b".to_string(), GgufValue::Bool(true)),
            ],
            tensors: vec![],
            alignment: 32,
            header_end: 0,
            tensor_data_offset: 0,
        };
        assert!(f.validate_format().is_empty());
    }

    #[test]
    fn encode_derives_counts_from_current_state() {
        let original = build_full_header_bytes();
        let mut f = GgufFile::parse(&original).expect("parse");
        f.metadata.push(("extra".to_string(), GgufValue::Uint8(7)));

        let encoded = f.encode_header();
        let f2 = GgufFile::parse(&encoded).expect("re-parse");
        assert_eq!(f2.metadata.len(), f.metadata.len());
        assert_eq!(f2.metadata.last().unwrap().0, "extra");
        assert_eq!(f2.metadata.last().unwrap().1, GgufValue::Uint8(7));
    }
}
