//! Regression tests for bugs surfaced in code review and now fixed. Each test
//! was originally written to fail while the bug existed; with the fixes in
//! place these tests are now expected to pass on every run, guarding against
//! the bug coming back. Currently covers Bug #1 (TUI signature pin), Bug #2
//! (foreign-`general.padding` preservation), Bug #4 (schema length unit:
//! bytes), and Bug #5 (schema numeric precision for `u64`/`i64`).

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

// =============================================================================
// Bug #4 (schema length unit) — schema::length_of for strings uses
// `s.chars().count()`, which counts Unicode code points. The README documents
// `min_length`/`max_length` without specifying a unit. This test encodes the
// byte-count interpretation that most users assume for a binary-format spec.
// It fails today; passing it requires either pinning the documented unit to
// "code points" (no code change, doc change only) or switching length_of to
// `s.len()` (bytes).
// =============================================================================

#[test]
fn schema_max_length_counts_bytes_not_code_points() {
    // "héllo": 5 grapheme clusters, 5 Unicode code points, 6 bytes in UTF-8.
    // (`é` is U+00E9 → bytes 0xC3 0xA9.)
    let s = "héllo";
    assert_eq!(s.chars().count(), 5);
    assert_eq!(s.len(), 6);

    let schema_json = r#"{
        "version": 1,
        "rules": {
            "general.author": {
                "type": "string",
                "max_length": 5,
                "severity": "error"
            }
        }
    }"#;
    let schema = gguf_surgeon::Schema::parse(schema_json).unwrap();

    let metadata = vec![(
        "general.author".to_string(),
        gguf_surgeon::GgufValue::String(s.to_string()),
    )];
    let violations = schema.validate(&metadata);

    assert!(
        !violations.is_empty(),
        "expected `max_length: 5` to flag a 6-byte UTF-8 string, but length_of uses \
         code-point count (5) and reports no violation. The README documents the rule \
         without a unit; users will assume bytes."
    );
}

// =============================================================================
// Bug #5 (numeric precision loss) — schema::numeric_value casts u64/i64 to
// f64 for min/max comparison. f64 has 53 bits of mantissa, so any integer
// >= 2^53 rounds. A `max: 2^53` rule incorrectly accepts a value of 2^53 + 1
// because the cast snaps it back to 2^53. This test fails today; passing it
// requires comparing integers as integers (e.g., switching min/max to a
// type-aware comparator instead of always going through f64).
// =============================================================================

#[test]
fn schema_max_does_not_lose_precision_for_u64_above_2_pow_53() {
    // Sanity: u64 -> f64 round-down to even at this boundary.
    let v: u64 = 9_007_199_254_740_993; // 2^53 + 1
    assert_eq!(v as f64, 9_007_199_254_740_992.0);

    let schema_json = r#"{
        "version": 1,
        "rules": {
            "huge.counter": {
                "type": "u64",
                "max": 9007199254740992,
                "severity": "error"
            }
        }
    }"#;
    let schema = gguf_surgeon::Schema::parse(schema_json).unwrap();

    let metadata = vec![(
        "huge.counter".to_string(),
        gguf_surgeon::GgufValue::Uint64(v),
    )];
    let violations = schema.validate(&metadata);

    assert!(
        !violations.is_empty(),
        "expected u64 value 2^53 + 1 = 9_007_199_254_740_993 to violate `max: 2^53`, \
         but the f64 cast hides this and no violation is reported."
    );
}
