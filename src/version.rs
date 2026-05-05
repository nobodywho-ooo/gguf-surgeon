pub const SUPPORTED_VERSIONS: &[u32] = &[3];

pub fn is_supported(v: u32) -> bool {
    SUPPORTED_VERSIONS.contains(&v)
}
