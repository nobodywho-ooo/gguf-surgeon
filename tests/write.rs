use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use gguf_surgeon::{GgufFile, GgufValue, GgufValueType, SavePath};

fn put_str(b: &mut Vec<u8>, s: &[u8]) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s);
}

fn put_kv_header(b: &mut Vec<u8>, key: &[u8], ty: GgufValueType) {
    put_str(b, key);
    b.extend_from_slice(&(ty as u32).to_le_bytes());
}

fn build_file_with_tensor_data() -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes()); // version
    b.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
    b.extend_from_slice(&2u64.to_le_bytes()); // kv_count

    put_kv_header(&mut b, b"general.architecture", GgufValueType::String);
    put_str(&mut b, b"llama");

    put_kv_header(&mut b, b"answer", GgufValueType::Uint32);
    b.extend_from_slice(&42u32.to_le_bytes());

    // tensor_info: w, dims=[4], type=0, offset=0
    put_str(&mut b, b"w");
    b.extend_from_slice(&1u32.to_le_bytes());
    b.extend_from_slice(&4u64.to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());

    // pad to default alignment 32
    let pad = (32 - b.len() % 32) % 32;
    b.extend(std::iter::repeat_n(0u8, pad));

    // tensor data: 64 bytes of a recognizable pattern
    for i in 0..64u8 {
        b.push(0xA0 ^ i);
    }
    b
}

static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn temp_path(name: &str) -> PathBuf {
    let n = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    std::env::temp_dir().join(format!("ggufmd-test-{pid}-{n}-{name}"))
}

struct Cleanup(Vec<PathBuf>);
impl Drop for Cleanup {
    fn drop(&mut self) {
        for p in &self.0 {
            let _ = std::fs::remove_file(p);
        }
    }
}

#[test]
fn write_modifies_metadata_and_preserves_tensor_data() {
    let src = temp_path("src.gguf");
    let dst = temp_path("dst.gguf");
    let _cleanup = Cleanup(vec![src.clone(), dst.clone(), tmp_for(&dst)]);

    let original_bytes = build_file_with_tensor_data();
    std::fs::write(&src, &original_bytes).unwrap();

    let mut f = GgufFile::read(&src).unwrap();
    // Mutate: change answer from 42 to 100, add a new key.
    let answer = f.metadata.iter_mut().find(|(k, _)| k == "answer").unwrap();
    answer.1 = GgufValue::Uint32(100);
    f.metadata
        .push(("new.flag".to_string(), GgufValue::Bool(true)));

    f.write(&src, &dst).unwrap();

    let f2 = GgufFile::read(&dst).unwrap();
    assert_eq!(
        f2.metadata
            .iter()
            .find(|(k, _)| k == "answer")
            .unwrap()
            .1,
        GgufValue::Uint32(100)
    );
    assert_eq!(
        f2.metadata
            .iter()
            .find(|(k, _)| k == "new.flag")
            .unwrap()
            .1,
        GgufValue::Bool(true)
    );

    // Tensor data preserved byte-for-byte.
    let new_bytes = std::fs::read(&dst).unwrap();
    let original_tensor_data = &original_bytes[f.tensor_data_offset as usize..];
    let new_tensor_data = &new_bytes[f2.tensor_data_offset as usize..];
    assert_eq!(original_tensor_data, new_tensor_data);
}

#[test]
fn write_in_place_replaces_original() {
    let path = temp_path("in-place.gguf");
    let _cleanup = Cleanup(vec![path.clone(), tmp_for(&path)]);

    std::fs::write(&path, build_file_with_tensor_data()).unwrap();

    let mut f = GgufFile::read(&path).unwrap();
    let answer = f.metadata.iter_mut().find(|(k, _)| k == "answer").unwrap();
    answer.1 = GgufValue::Uint32(7);

    f.write(&path, &path).unwrap();

    let f2 = GgufFile::read(&path).unwrap();
    assert_eq!(
        f2.metadata.iter().find(|(k, _)| k == "answer").unwrap().1,
        GgufValue::Uint32(7)
    );
    // Temp file is gone after a successful in-place write.
    assert!(!tmp_for(&path).exists());
}

fn tmp_for(p: &PathBuf) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(".tmp");
    PathBuf::from(s)
}

#[test]
fn first_save_grows_file_to_install_padding() {
    let path = temp_path("first-save.gguf");
    let _cleanup = Cleanup(vec![path.clone(), tmp_for(&path)]);

    std::fs::write(&path, build_file_with_tensor_data()).unwrap();
    let original_size = std::fs::metadata(&path).unwrap().len();

    let mut f = GgufFile::read(&path).unwrap();
    // Even a same-size edit triggers a FullRewrite on first save because the padding
    // key is being added (file grows from minimum-aligned to slack-aligned).
    let answer = f.metadata.iter_mut().find(|(k, _)| k == "answer").unwrap();
    answer.1 = GgufValue::Uint32(7);

    assert_eq!(f.pick_save_path(), SavePath::FullRewrite);
    f.write(&path, &path).unwrap();

    let new_size = std::fs::metadata(&path).unwrap().len();
    assert!(new_size >= 64 * 1024);
    assert!(new_size > original_size);
}

#[test]
fn padded_file_uses_header_overwrite_for_subsequent_same_size_edits() {
    let path = temp_path("subsequent.gguf");
    let _cleanup = Cleanup(vec![path.clone(), tmp_for(&path)]);

    std::fs::write(&path, build_file_with_tensor_data()).unwrap();

    // First save installs the padding.
    let mut f1 = GgufFile::read(&path).unwrap();
    f1.write(&path, &path).unwrap();
    let padded_size = std::fs::metadata(&path).unwrap().len();

    // Second save with same-size edit must stay header-overwrite.
    let mut f2 = GgufFile::read(&path).unwrap();
    let original_tdo = f2.tensor_data_offset;
    let answer = f2.metadata.iter_mut().find(|(k, _)| k == "answer").unwrap();
    answer.1 = GgufValue::Uint32(7);

    assert_eq!(f2.pick_save_path(), SavePath::HeaderOverwrite);
    f2.write(&path, &path).unwrap();

    let f3 = GgufFile::read(&path).unwrap();
    assert_eq!(f3.tensor_data_offset, original_tdo);
    assert_eq!(std::fs::metadata(&path).unwrap().len(), padded_size);
    assert_eq!(
        f3.metadata.iter().find(|(k, _)| k == "answer").unwrap().1,
        GgufValue::Uint32(7)
    );
}

#[test]
fn padded_file_uses_header_overwrite_for_small_size_changing_edits() {
    let path = temp_path("small-change.gguf");
    let _cleanup = Cleanup(vec![path.clone(), tmp_for(&path)]);

    std::fs::write(&path, build_file_with_tensor_data()).unwrap();

    // First save installs padding.
    let mut f1 = GgufFile::read(&path).unwrap();
    f1.write(&path, &path).unwrap();
    let padded_size = std::fs::metadata(&path).unwrap().len();

    // Add a small key — fits inside the 64 KB slack budget, should stay header-overwrite.
    let mut f2 = GgufFile::read(&path).unwrap();
    f2.metadata
        .push(("small.new.key".to_string(), GgufValue::Uint32(123)));

    assert_eq!(f2.pick_save_path(), SavePath::HeaderOverwrite);
    f2.write(&path, &path).unwrap();

    let f3 = GgufFile::read(&path).unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().len(), padded_size);
    assert_eq!(
        f3.metadata
            .iter()
            .find(|(k, _)| k == "small.new.key")
            .unwrap()
            .1,
        GgufValue::Uint32(123)
    );
}
