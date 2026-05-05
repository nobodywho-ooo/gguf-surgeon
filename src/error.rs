use std::io;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] io::Error),

    #[error("not a GGUF file: bad magic {0:?}")]
    BadMagic([u8; 4]),

    #[error("unsupported GGUF version {found}; this build supports {supported:?}")]
    UnsupportedVersion {
        found: u32,
        supported: &'static [u32],
    },

    #[error("unexpected end of file at offset {offset}")]
    Truncated { offset: u64 },

    #[error("unknown value type tag {0}")]
    UnknownValueType(u32),

    #[error("string is not valid UTF-8 at offset {offset}")]
    InvalidUtf8 { offset: u64 },

    #[error("declared size {declared} exceeds remaining file bytes {remaining}")]
    OversizedDeclaration { declared: u64, remaining: u64 },
}
