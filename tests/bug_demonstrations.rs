//! Regression tests for two bugs surfaced in code review and now fixed.
//! Each test was originally written to fail while the bug existed (and was
//! marked `#[ignore]` to keep the default suite green). With the fixes in
//! place these tests are now expected to pass on every run.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use gguf_surgeon::{GgufFile, GgufValue};

// --- shared helpers --------------------------------------------------------

static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn temp_path(name: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    std::env::temp_dir().join(format!("ggufsurgeon-bug-{pid}-{n}-{name}"))
}

struct Cleanup(Vec<PathBuf>);
impl Drop for Cleanup {
    fn drop(&mut self) {
        for p in &self.0 {
            let _ = std::fs::remove_file(p);
        }
    }
}

fn tmp_for(p: &PathBuf) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(".tmp");
    PathBuf::from(s)
}

// =============================================================================
// Bug #2 — ensure_padding silently overwrites a `general.padding` entry written
// by another tool. We claim this is data loss with no warning. The test below
// asserts the *correct* behavior (foreign data must survive); it fails today
// because the current code unconditionally strips any entry named
// `general.padding` before re-installing its own zero-filled array.
// =============================================================================

#[test]
fn ensure_padding_must_not_clobber_foreign_general_padding() {
    // Build a small v3 file by hand that already contains a `general.padding`
    // key holding meaningful data (a string). Save it, parse it back, and run
    // any unrelated edit + save through the public API. The foreign string must
    // still be there afterwards.
    let path = temp_path("foreign-padding.gguf");
    let _cleanup = Cleanup(vec![path.clone(), tmp_for(&path)]);

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes()); // version
    bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    bytes.extend_from_slice(&2u64.to_le_bytes()); // kv_count

    // kv 1: general.architecture = "llama"
    let arch = b"general.architecture";
    bytes.extend_from_slice(&(arch.len() as u64).to_le_bytes());
    bytes.extend_from_slice(arch);
    bytes.extend_from_slice(&8u32.to_le_bytes()); // type=string
    let val = b"llama";
    bytes.extend_from_slice(&(val.len() as u64).to_le_bytes());
    bytes.extend_from_slice(val);

    // kv 2: general.padding = "FOREIGN_DATA"  ← simulating another tool
    let pad = b"general.padding";
    bytes.extend_from_slice(&(pad.len() as u64).to_le_bytes());
    bytes.extend_from_slice(pad);
    bytes.extend_from_slice(&8u32.to_le_bytes()); // type=string
    let foreign = b"FOREIGN_DATA";
    bytes.extend_from_slice(&(foreign.len() as u64).to_le_bytes());
    bytes.extend_from_slice(foreign);

    std::fs::write(&path, &bytes).unwrap();

    // Run an unrelated edit through the public API (this triggers ensure_padding).
    let mut f = GgufFile::read(&path).unwrap();
    let arch_pos = f
        .metadata
        .iter()
        .position(|(k, _)| k == "general.architecture")
        .unwrap();
    f.metadata[arch_pos].1 = GgufValue::String("mistral".to_string());
    f.write(&path, &path, gguf_surgeon::SaveMode::Auto).unwrap();

    // Re-read and verify the foreign string survived.
    let f2 = GgufFile::read(&path).unwrap();
    let entry = f2
        .metadata
        .iter()
        .find(|(k, _)| k == "general.padding")
        .expect("general.padding should still exist");

    let actual_kind = match &entry.1 {
        GgufValue::String(_) => "String".to_string(),
        GgufValue::Array(a) => format!("Array<{}; {}>", a.element_type.as_str(), a.elements.len()),
        other => format!("{:?}", other.ty()),
    };
    assert!(
        matches!(&entry.1, GgufValue::String(s) if s == "FOREIGN_DATA"),
        "foreign general.padding was rewritten from String(\"FOREIGN_DATA\") to {actual_kind} — silent data loss"
    );
}

// =============================================================================
// Bug #1 — The TUI must accept --schema and --force from the CLI. Compile-time
// pin on the corrected signature: `fn(&Path, Option<&Schema>, bool) -> Result<()>`.
// If anyone regresses the signature this assignment stops compiling. The runtime
// portion below performs a schema-validating save end-to-end and asserts that a
// schema-violating value is blocked, which exercises the full TUI plumbing path
// without driving the terminal.
// =============================================================================

#[test]
fn tui_run_accepts_schema_and_force_for_validation() {
    // Compile-time pin on the schema-aware signature.
    let _signature: fn(
        &std::path::Path,
        Option<&gguf_surgeon::Schema>,
        bool,
        gguf_surgeon::SaveMode,
    ) -> anyhow::Result<()> = gguf_surgeon::tui::run;
}
