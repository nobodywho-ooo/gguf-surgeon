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

/// Vendor-namespaced sentinel key holding zero-filled padding bytes. Using a
/// non-spec namespace prevents collisions with metadata that other tools might
/// legitimately store under a `general.*` name.
const PADDING_KEY: &str = "ggufsurgeon.padding";

/// Older versions of this editor used `general.padding`. On save we migrate by
/// stripping any entry under that key whose content looks like ours (a u8 array
/// of all zeros) — but not foreign content under the same key, which we keep.
const OLD_PADDING_KEY: &str = "general.padding";

/// Default slack step for the sentinel padding key. The header is grown to a multiple of
/// this value so subsequent edits within the budget can use the header-overwrite save path.
pub const DEFAULT_PADDING_STEP: u64 = 64 * 1024;

/// Length in bytes of "ggufsurgeon.padding" (the sentinel key name).
const PADDING_KEY_LEN: u64 = 19;

/// Encoded byte overhead of the sentinel padding key itself, given the file version.
/// (key length prefix + key bytes + value type tag + array element type + array length prefix)
fn padding_overhead(version: u32) -> u64 {
    let cp = version::count_prefix_bytes(version);
    cp + PADDING_KEY_LEN + 4 + 4 + cp
}

#[derive(Debug, Clone)]
pub struct GgufFile {
    pub version: u32,
    /// File endianness. GGUF v3 added big-endian support; we detect at parse time
    /// by reading the version field in both byte orders and picking the one whose
    /// value falls in the supported set. Preserved on save so files round-trip in
    /// their original byte order.
    pub little_endian: bool,
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

/// Cap an attacker-controlled count so `Vec::with_capacity` cannot be tricked into
/// allocating gigabytes from a malformed file. The Vec grows naturally as elements
/// are pushed; the bounds check inside the per-element read terminates the loop on
/// real truncation, so total memory stays bounded by what's actually in the file.
const SAFE_CAPACITY_HINT: usize = 4096;

fn safe_capacity(declared: u64) -> usize {
    declared.min(SAFE_CAPACITY_HINT as u64) as usize
}

/// Reserved metadata keys that the editor manages internally. Users cannot edit them
/// directly; they are filtered out of user-facing displays.
pub fn is_reserved_key(key: &str) -> bool {
    key == PADDING_KEY
}

/// Tokenizer vocabulary array key.
const TOKENS_KEY: &str = "tokenizer.ggml.tokens";

/// Special-token-id metadata keys that, when present, must point to valid indices
/// in `tokenizer.ggml.tokens`. Out-of-range or negative values pass parsing but
/// crash generation in `llama.cpp`.
const TOKEN_ID_KEYS: &[&str] = &[
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.eot_token_id",
    "tokenizer.ggml.eom_token_id",
    "tokenizer.ggml.unknown_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.cls_token_id",
    "tokenizer.ggml.mask_token_id",
    "tokenizer.ggml.separator_token_id",
];

/// Coerce a `GgufValue` to an i64 if it's an integer type. Returns None for
/// non-integer types or unsigned values larger than i64::MAX.
fn as_signed_index(v: &GgufValue) -> Option<i64> {
    match v {
        GgufValue::Uint8(n) => Some(i64::from(*n)),
        GgufValue::Uint16(n) => Some(i64::from(*n)),
        GgufValue::Uint32(n) => Some(i64::from(*n)),
        GgufValue::Uint64(n) => i64::try_from(*n).ok(),
        GgufValue::Int8(n) => Some(i64::from(*n)),
        GgufValue::Int16(n) => Some(i64::from(*n)),
        GgufValue::Int32(n) => Some(i64::from(*n)),
        GgufValue::Int64(n) => Some(*n),
        _ => None,
    }
}

/// Detect a value that this editor (current or older) wrote as padding: a u8 array
/// where every element is zero. Used at save time to decide whether the legacy
/// `general.padding` key holds our slack or foreign data we must preserve.
fn looks_like_our_padding(value: &GgufValue) -> bool {
    let GgufValue::Array(a) = value else {
        return false;
    };
    if a.element_type != GgufValueType::Uint8 {
        return false;
    }
    a.elements.iter().all(|e| matches!(e, GgufValue::Uint8(0)))
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

        // Detect endianness from the version field: try LE first, then BE. The first
        // interpretation that lands on a known version wins. GGUF v3 added formal
        // big-endian support; v1/v2 are accepted in either byte order for permissiveness.
        let version_bytes: [u8; 4] = r.take(4)?.try_into().unwrap();
        let v_le = u32::from_le_bytes(version_bytes);
        let v_be = u32::from_be_bytes(version_bytes);
        let (version, little_endian) = if version::is_supported(v_le) {
            (v_le, true)
        } else if version::is_supported(v_be) {
            (v_be, false)
        } else {
            return Err(Error::UnsupportedVersion {
                found: v_le,
                supported: version::SUPPORTED_VERSIONS,
            });
        };
        r.version = version;
        r.little_endian = little_endian;

        let tensor_count = r.count()?;
        let kv_count = r.count()?;

        if kv_count > r.remaining() as u64 {
            return Err(Error::OversizedDeclaration {
                declared: kv_count,
                remaining: r.remaining() as u64,
            });
        }

        let mut metadata = Vec::with_capacity(safe_capacity(kv_count));
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
        let mut tensors = Vec::with_capacity(safe_capacity(tensor_count));
        for _ in 0..tensor_count {
            tensors.push(r.tensor_info()?);
        }

        let alignment = alignment_from_metadata(&metadata);

        let header_end = r.pos() as u64;
        let tensor_data_offset = align_up(header_end, alignment);

        Ok(GgufFile {
            version,
            little_endian,
            tensor_count,
            metadata,
            tensors,
            alignment,
            header_end,
            tensor_data_offset,
        })
    }

    /// Insert (or replace) the `ggufsurgeon.padding` sentinel key so the encoded header
    /// rounds up to a multiple of `step` bytes. This reserves slack for future small
    /// edits, which can then be saved via the header-overwrite path without copying
    /// tensor data. Pass `step = 0` to remove any existing padding key.
    ///
    /// Migration: any `general.padding` entry from a previous version of this editor
    /// is stripped only when its content matches "ours" (a u8 array of all zeros).
    /// Foreign data under the same key is left untouched — fixing the silent-data-loss
    /// bug from earlier versions.
    pub fn ensure_padding(&mut self, step: u64) {
        self.metadata.retain(|(k, v)| {
            if k == PADDING_KEY {
                return false; // always strip our own sentinel; we'll re-add it
            }
            if k == OLD_PADDING_KEY && looks_like_our_padding(v) {
                return false; // migrate: strip old-style padding we wrote ourselves
            }
            true
        });
        if step == 0 {
            return;
        }
        let raw = self.encoded_size_unaligned();
        let overhead = padding_overhead(self.version);
        let min_total_with_padding = raw + overhead;
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
        let cp = version::count_prefix_bytes(self.version);
        let mut size: u64 = 4 + 4 + cp + cp; // magic + version + tensor_count + kv_count
        for (k, v) in &self.metadata {
            size += cp + k.len() as u64 + 4 + value_encoded_size(v, self.version);
        }
        for t in &self.tensors {
            size += cp + t.name.len() as u64 + 4 + 8 * t.dims.len() as u64 + 4 + 8;
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

        // Token-id range check. When tokenizer.ggml.tokens is present and the file
        // declares a special token id (BOS/EOS/UNK/PAD/...) that is negative or
        // points past the end of the tokens array, llama.cpp loads the file but
        // crashes at generation time. Catching it at save-time matches the README's
        // "format-level → unconditional block" promise.
        if let Some(tokens_len) = self
            .metadata
            .iter()
            .find(|(k, _)| k == TOKENS_KEY)
            .and_then(|(_, v)| match v {
                GgufValue::Array(a) => Some(a.elements.len() as u64),
                _ => None,
            })
        {
            for &key in TOKEN_ID_KEYS {
                if let Some((_, value)) = self.metadata.iter().find(|(k, _)| k == key) {
                    match as_signed_index(value) {
                        Some(id) if id < 0 => out.push(Violation {
                            origin: Origin::Format,
                            key: key.to_string(),
                            severity: Severity::Error,
                            message: format!("token id {id} is negative"),
                        }),
                        Some(id) if (id as u64) >= tokens_len => out.push(Violation {
                            origin: Origin::Format,
                            key: key.to_string(),
                            severity: Severity::Error,
                            message: format!(
                                "token id {id} is out of range (tokens array has {tokens_len} elements)"
                            ),
                        }),
                        _ => {}
                    }
                }
            }
        }

        out
    }

    /// Return Err if format-level validation finds any error-severity violations.
    /// Used internally by `write()` to refuse producing an invalid file.
    pub fn check_format(&self) -> Result<(), Error> {
        let violations = self.validate_format();
        let errors: Vec<_> = violations
            .iter()
            .filter(|v| v.severity == Severity::Error)
            .collect();
        if errors.is_empty() {
            return Ok(());
        }
        let summary = errors
            .iter()
            .map(|v| format!("{}: {}", v.key, v.message))
            .collect::<Vec<_>>()
            .join("; ");
        Err(Error::FormatViolation(summary))
    }

    /// Encode the header region (magic through minimum alignment padding) into a byte vector.
    /// Counts and alignment padding are derived from the current state, not from stale fields.
    /// Reserving extra slack for future in-place edits is a save-path concern (sentinel key
    /// or alignment adjustment) and is intentionally not handled here.
    pub fn encode_header(&self) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut w = Writer::new(&mut out, self.version, self.little_endian);
            w.bytes(MAGIC);
            w.u32(self.version);
            w.count(self.tensors.len() as u64);
            w.count(self.metadata.len() as u64);
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
    /// 0 until the version field is read; then the file's declared version.
    /// Determines whether length prefixes are u32 (v1) or u64 (v2/v3).
    version: u32,
    /// Set after detecting endianness from the version field. Determines whether
    /// multi-byte primitives are decoded little- or big-endian.
    little_endian: bool,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            version: 0,
            little_endian: true,
        }
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

    fn count(&mut self) -> Result<u64, Error> {
        if self.version == 1 {
            Ok(u64::from(self.u32()?))
        } else {
            self.u64()
        }
    }

    fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.take(1)?[0])
    }
    fn i8(&mut self) -> Result<i8, Error> {
        Ok(self.take(1)?[0] as i8)
    }
    fn u16(&mut self) -> Result<u16, Error> {
        let b: [u8; 2] = self.take(2)?.try_into().unwrap();
        Ok(if self.little_endian { u16::from_le_bytes(b) } else { u16::from_be_bytes(b) })
    }
    fn i16(&mut self) -> Result<i16, Error> {
        let b: [u8; 2] = self.take(2)?.try_into().unwrap();
        Ok(if self.little_endian { i16::from_le_bytes(b) } else { i16::from_be_bytes(b) })
    }
    fn u32(&mut self) -> Result<u32, Error> {
        let b: [u8; 4] = self.take(4)?.try_into().unwrap();
        Ok(if self.little_endian { u32::from_le_bytes(b) } else { u32::from_be_bytes(b) })
    }
    fn i32(&mut self) -> Result<i32, Error> {
        let b: [u8; 4] = self.take(4)?.try_into().unwrap();
        Ok(if self.little_endian { i32::from_le_bytes(b) } else { i32::from_be_bytes(b) })
    }
    fn u64(&mut self) -> Result<u64, Error> {
        let b: [u8; 8] = self.take(8)?.try_into().unwrap();
        Ok(if self.little_endian { u64::from_le_bytes(b) } else { u64::from_be_bytes(b) })
    }
    fn i64(&mut self) -> Result<i64, Error> {
        let b: [u8; 8] = self.take(8)?.try_into().unwrap();
        Ok(if self.little_endian { i64::from_le_bytes(b) } else { i64::from_be_bytes(b) })
    }
    fn f32(&mut self) -> Result<f32, Error> {
        let b: [u8; 4] = self.take(4)?.try_into().unwrap();
        Ok(if self.little_endian { f32::from_le_bytes(b) } else { f32::from_be_bytes(b) })
    }
    fn f64(&mut self) -> Result<f64, Error> {
        let b: [u8; 8] = self.take(8)?.try_into().unwrap();
        Ok(if self.little_endian { f64::from_le_bytes(b) } else { f64::from_be_bytes(b) })
    }
    fn bool_(&mut self) -> Result<bool, Error> {
        Ok(self.u8()? != 0)
    }

    fn string(&mut self) -> Result<String, Error> {
        let len = self.count()?;
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
        let len = self.count()?;
        if len > self.remaining() as u64 {
            return Err(Error::OversizedDeclaration {
                declared: len,
                remaining: self.remaining() as u64,
            });
        }
        let mut elements = Vec::with_capacity(safe_capacity(len));
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
        let mut dims = Vec::with_capacity(safe_capacity(n_dims));
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

fn value_encoded_size(v: &GgufValue, version: u32) -> u64 {
    let cp = version::count_prefix_bytes(version);
    match v {
        GgufValue::Uint8(_) | GgufValue::Int8(_) | GgufValue::Bool(_) => 1,
        GgufValue::Uint16(_) | GgufValue::Int16(_) => 2,
        GgufValue::Uint32(_) | GgufValue::Int32(_) | GgufValue::Float32(_) => 4,
        GgufValue::Uint64(_) | GgufValue::Int64(_) | GgufValue::Float64(_) => 8,
        GgufValue::String(s) => cp + s.len() as u64,
        GgufValue::Array(a) => 4 + cp + a.elements.iter().map(|e| value_encoded_size(e, version)).sum::<u64>(),
    }
}

struct Writer<'a> {
    out: &'a mut Vec<u8>,
    version: u32,
    little_endian: bool,
}

impl<'a> Writer<'a> {
    fn new(out: &'a mut Vec<u8>, version: u32, little_endian: bool) -> Self {
        Self { out, version, little_endian }
    }

    fn count(&mut self, n: u64) {
        if self.version == 1 {
            let n32: u32 = n.try_into().expect("v1 count fits in u32");
            self.u32(n32);
        } else {
            self.u64(n);
        }
    }

    fn bytes(&mut self, b: &[u8]) {
        self.out.extend_from_slice(b);
    }
    fn u8(&mut self, v: u8) {
        self.out.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.out.extend_from_slice(&if self.little_endian { v.to_le_bytes() } else { v.to_be_bytes() });
    }
    fn u32(&mut self, v: u32) {
        self.out.extend_from_slice(&if self.little_endian { v.to_le_bytes() } else { v.to_be_bytes() });
    }
    fn u64(&mut self, v: u64) {
        self.out.extend_from_slice(&if self.little_endian { v.to_le_bytes() } else { v.to_be_bytes() });
    }
    fn i8_(&mut self, v: i8) {
        self.out.push(v as u8);
    }
    fn i16(&mut self, v: i16) {
        self.out.extend_from_slice(&if self.little_endian { v.to_le_bytes() } else { v.to_be_bytes() });
    }
    fn i32(&mut self, v: i32) {
        self.out.extend_from_slice(&if self.little_endian { v.to_le_bytes() } else { v.to_be_bytes() });
    }
    fn i64(&mut self, v: i64) {
        self.out.extend_from_slice(&if self.little_endian { v.to_le_bytes() } else { v.to_be_bytes() });
    }
    fn f32(&mut self, v: f32) {
        self.out.extend_from_slice(&if self.little_endian { v.to_le_bytes() } else { v.to_be_bytes() });
    }
    fn f64(&mut self, v: f64) {
        self.out.extend_from_slice(&if self.little_endian { v.to_le_bytes() } else { v.to_be_bytes() });
    }
    fn bool_(&mut self, v: bool) {
        self.out.push(u8::from(v));
    }

    fn string(&mut self, s: &str) {
        self.count(s.len() as u64);
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
                self.count(a.elements.len() as u64);
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
            little_endian: true,
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
            little_endian: true,
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
    fn safe_capacity_caps_attacker_controlled_count() {
        assert_eq!(safe_capacity(0), 0);
        assert_eq!(safe_capacity(100), 100);
        assert_eq!(safe_capacity(SAFE_CAPACITY_HINT as u64), SAFE_CAPACITY_HINT);
        // The headline case: a malformed file declaring billions of elements must not be
        // allowed to request gigabytes of pre-allocation.
        assert_eq!(safe_capacity(1_000_000_000), SAFE_CAPACITY_HINT);
        assert_eq!(safe_capacity(u64::MAX), SAFE_CAPACITY_HINT);
    }

    #[test]
    fn parser_handles_huge_declared_count_without_pre_allocating() {
        // Declare a giant kv_count (1M) but provide only a handful of bytes after the header.
        // The parser must error on truncation, not allocate proportional to the declaration.
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        b.extend_from_slice(&1_000_000u64.to_le_bytes()); // kv_count: pretend we have 1M entries
        // ... but no entries follow.
        let err = GgufFile::parse(&b).unwrap_err();
        assert!(matches!(err, Error::OversizedDeclaration { .. } | Error::Truncated { .. }));
    }

    #[test]
    fn is_reserved_key_recognises_padding() {
        assert!(is_reserved_key(PADDING_KEY));
        assert!(is_reserved_key("ggufsurgeon.padding"));
        // `general.padding` is no longer reserved — see Bug #2 fix; users with
        // foreign data under that key can edit it freely.
        assert!(!is_reserved_key("general.padding"));
        assert!(!is_reserved_key("general.architecture"));
        assert!(!is_reserved_key(""));
        assert!(!is_reserved_key("ggufsurgeon.padding.suffix"));
    }

    #[test]
    fn check_format_returns_error_for_duplicate_keys() {
        let f = GgufFile {
            version: 3,
            little_endian: true,
            tensor_count: 0,
            metadata: vec![
                ("a".to_string(), GgufValue::Uint32(1)),
                ("a".to_string(), GgufValue::Uint32(2)),
            ],
            tensors: vec![],
            alignment: 32,
            header_end: 0,
            tensor_data_offset: 0,
        };
        let err = f.check_format().unwrap_err();
        assert!(matches!(err, Error::FormatViolation(_)));
    }

    #[test]
    fn check_format_passes_clean_file() {
        let f = GgufFile {
            version: 3,
            little_endian: true,
            tensor_count: 0,
            metadata: vec![("a".to_string(), GgufValue::Uint32(1))],
            tensors: vec![],
            alignment: 32,
            header_end: 0,
            tensor_data_offset: 0,
        };
        assert!(f.check_format().is_ok());
    }

    fn header_v(version: u32, tensor_count: u64, kv_count: u64) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&version.to_le_bytes());
        if version == 1 {
            b.extend_from_slice(&(tensor_count as u32).to_le_bytes());
            b.extend_from_slice(&(kv_count as u32).to_le_bytes());
        } else {
            b.extend_from_slice(&tensor_count.to_le_bytes());
            b.extend_from_slice(&kv_count.to_le_bytes());
        }
        b
    }

    fn put_str_v(b: &mut Vec<u8>, s: &[u8], version: u32) {
        if version == 1 {
            b.extend_from_slice(&(s.len() as u32).to_le_bytes());
        } else {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        }
        b.extend_from_slice(s);
    }

    #[test]
    fn parses_v2_file_with_same_layout_as_v3() {
        let mut b = header_v(2, 0, 1);
        put_str(&mut b, b"answer");
        b.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
        b.extend_from_slice(&42u32.to_le_bytes());
        let f = GgufFile::parse(&b).expect("parse v2");
        assert_eq!(f.version, 2);
        assert_eq!(f.metadata.len(), 1);
        assert_eq!(f.metadata[0].1, GgufValue::Uint32(42));

        // Round-trip: v2 file encodes back as v2.
        let encoded = f.encode_header();
        let f2 = GgufFile::parse(&encoded).expect("re-parse v2");
        assert_eq!(f2.version, 2);
        assert_eq!(f2.metadata, f.metadata);
    }

    #[test]
    fn parses_v1_file_with_u32_length_prefixes() {
        let mut b = header_v(1, 0, 2);

        // kv 1: ("name", string, "llama")
        put_str_v(&mut b, b"name", 1);
        b.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        put_str_v(&mut b, b"llama", 1);

        // kv 2: ("ctx", uint32, 4096)
        put_str_v(&mut b, b"ctx", 1);
        b.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
        b.extend_from_slice(&4096u32.to_le_bytes());

        let f = GgufFile::parse(&b).expect("parse v1");
        assert_eq!(f.version, 1);
        assert_eq!(f.metadata.len(), 2);
        assert_eq!(f.metadata[0].0, "name");
        assert_eq!(f.metadata[0].1, GgufValue::String("llama".to_string()));
        assert_eq!(f.metadata[1].1, GgufValue::Uint32(4096));
    }

    #[test]
    fn v1_round_trip_byte_identical() {
        let mut original = header_v(1, 0, 1);
        put_str_v(&mut original, b"k", 1);
        original.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
        original.extend_from_slice(&7u32.to_le_bytes());
        // Pad to default alignment 32.
        let pad = (32 - original.len() % 32) % 32;
        original.extend(std::iter::repeat_n(0u8, pad));

        let f = GgufFile::parse(&original).expect("parse");
        let encoded = f.encode_header();
        assert_eq!(encoded, original, "v1 round-trip must preserve byte layout");
    }

    #[test]
    fn v1_array_uses_u32_length_prefix() {
        let mut b = header_v(1, 0, 1);
        put_str_v(&mut b, b"shape", 1);
        b.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
        b.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes()); // element type
        b.extend_from_slice(&3u32.to_le_bytes()); // length, u32 in v1
        b.extend_from_slice(&10u32.to_le_bytes());
        b.extend_from_slice(&20u32.to_le_bytes());
        b.extend_from_slice(&30u32.to_le_bytes());

        let f = GgufFile::parse(&b).expect("parse v1 array");
        let GgufValue::Array(arr) = &f.metadata[0].1 else {
            panic!("expected array");
        };
        assert_eq!(arr.elements.len(), 3);
        assert_eq!(arr.elements[2], GgufValue::Uint32(30));
    }

    fn file_with_tokens_and_special_id(
        n_tokens: usize,
        special_key: &str,
        id: GgufValue,
    ) -> GgufFile {
        let elements = (0..n_tokens)
            .map(|i| GgufValue::String(format!("tok{i}")))
            .collect::<Vec<_>>();
        GgufFile {
            version: 3,
            little_endian: true,
            tensor_count: 0,
            metadata: vec![
                (
                    TOKENS_KEY.to_string(),
                    GgufValue::Array(GgufArray {
                        element_type: GgufValueType::String,
                        elements,
                    }),
                ),
                (special_key.to_string(), id),
            ],
            tensors: vec![],
            alignment: 32,
            header_end: 0,
            tensor_data_offset: 0,
        }
    }

    #[test]
    fn validate_format_passes_in_range_token_id() {
        let f = file_with_tokens_and_special_id(
            128,
            "tokenizer.ggml.eos_token_id",
            GgufValue::Uint32(42),
        );
        assert!(f.validate_format().is_empty());
    }

    #[test]
    fn validate_format_flags_out_of_range_token_id() {
        let f = file_with_tokens_and_special_id(
            128,
            "tokenizer.ggml.eos_token_id",
            GgufValue::Uint32(500),
        );
        let v = f.validate_format();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].origin, Origin::Format);
        assert_eq!(v[0].severity, Severity::Error);
        assert!(v[0].message.contains("out of range"));
    }

    #[test]
    fn validate_format_flags_negative_token_id() {
        let f = file_with_tokens_and_special_id(
            128,
            "tokenizer.ggml.bos_token_id",
            GgufValue::Int32(-1),
        );
        let v = f.validate_format();
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("negative"));
    }

    #[test]
    fn validate_format_skips_token_id_check_without_tokens_array() {
        let f = GgufFile {
            version: 3,
            little_endian: true,
            tensor_count: 0,
            metadata: vec![(
                "tokenizer.ggml.eos_token_id".to_string(),
                GgufValue::Uint32(99999),
            )],
            tensors: vec![],
            alignment: 32,
            header_end: 0,
            tensor_data_offset: 0,
        };
        // Without the tokens array we can't check; this should NOT produce a
        // violation (we're permissive about partial tokenizer metadata).
        assert!(f.validate_format().is_empty());
    }

    #[test]
    fn parses_big_endian_v3_file() {
        // Hand-build a v3 BE file: same layout as LE but every multi-byte integer
        // is in big-endian byte order.
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_be_bytes());
        b.extend_from_slice(&0u64.to_be_bytes()); // tensor_count
        b.extend_from_slice(&1u64.to_be_bytes()); // kv_count
        // kv: ("answer", uint32, 42)
        let key = b"answer";
        b.extend_from_slice(&(key.len() as u64).to_be_bytes());
        b.extend_from_slice(key);
        b.extend_from_slice(&(GgufValueType::Uint32 as u32).to_be_bytes());
        b.extend_from_slice(&42u32.to_be_bytes());

        let f = GgufFile::parse(&b).expect("parse BE");
        assert_eq!(f.version, 3);
        assert!(!f.little_endian, "endianness must be detected as big-endian");
        assert_eq!(f.metadata.len(), 1);
        assert_eq!(f.metadata[0].0, "answer");
        assert_eq!(f.metadata[0].1, GgufValue::Uint32(42));
    }

    #[test]
    fn big_endian_round_trip_preserves_byte_order() {
        let mut original = Vec::new();
        original.extend_from_slice(b"GGUF");
        original.extend_from_slice(&3u32.to_be_bytes());
        original.extend_from_slice(&0u64.to_be_bytes());
        original.extend_from_slice(&1u64.to_be_bytes());
        let key = b"k";
        original.extend_from_slice(&(key.len() as u64).to_be_bytes());
        original.extend_from_slice(key);
        original.extend_from_slice(&(GgufValueType::Uint32 as u32).to_be_bytes());
        original.extend_from_slice(&7u32.to_be_bytes());
        let pad = (32 - original.len() % 32) % 32;
        original.extend(std::iter::repeat_n(0u8, pad));

        let f = GgufFile::parse(&original).expect("parse BE");
        assert!(!f.little_endian);
        let encoded = f.encode_header();
        assert_eq!(
            encoded, original,
            "BE round-trip must preserve byte-for-byte layout"
        );

        // And the re-parsed file must still report BE.
        let f2 = GgufFile::parse(&encoded).expect("re-parse BE");
        assert!(!f2.little_endian);
        assert_eq!(f2.metadata, f.metadata);
    }

    #[test]
    fn endianness_detection_falls_back_correctly() {
        // A v3 LE file's version field reads as 3 in LE. The BE interpretation would
        // be 50_331_648 (0x03000000) — not in SUPPORTED_VERSIONS, so the parser must
        // pick LE.
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        let f = GgufFile::parse(&b).expect("parse LE");
        assert!(f.little_endian);
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
