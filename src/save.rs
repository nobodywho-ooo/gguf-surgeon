use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::error::Error;
use crate::format::{DEFAULT_PADDING_STEP, GgufFile};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SavePath {
    /// Header is the same size as the original; clone the source to a temp via copy-on-write
    /// (where the filesystem supports it) and overwrite only the header region.
    HeaderOverwrite,
    /// Header changed size; stream tensor data through a fresh temp file.
    FullRewrite,
}

impl GgufFile {
    /// Predict which save path `write()` will take for the current state at the given
    /// padding step. Same encoded header size as the original `tensor_data_offset` → cheap
    /// header overwrite (CoW clone on APFS/Btrfs/XFS-with-reflink). Different size → full
    /// streaming rewrite.
    pub fn predict_save_path(&self, padding_step: u64) -> SavePath {
        let mut tmp = self.clone();
        tmp.ensure_padding(padding_step);
        if tmp.encode_header().len() as u64 == self.tensor_data_offset {
            SavePath::HeaderOverwrite
        } else {
            SavePath::FullRewrite
        }
    }

    /// `predict_save_path` using the default padding step.
    pub fn pick_save_path(&self) -> SavePath {
        self.predict_save_path(DEFAULT_PADDING_STEP)
    }

    /// Write this file's header plus the tensor data from `source` into `dest`.
    /// Atomic: writes to `<dest>.tmp`, fsyncs, then renames over `dest`.
    /// Also adjusts the `general.padding` sentinel so the encoded header rounds up to the
    /// default 64 KB slack budget — subsequent small edits then take the header-overwrite path.
    pub fn write(&mut self, source: &Path, dest: &Path) -> Result<(), Error> {
        self.ensure_padding(DEFAULT_PADDING_STEP);
        let header = self.encode_header();
        if header.len() as u64 == self.tensor_data_offset {
            self.write_header_overwrite(source, dest, &header)
        } else {
            self.write_full_rewrite(source, dest, &header)
        }
    }

    fn write_header_overwrite(&self, source: &Path, dest: &Path, header: &[u8]) -> Result<(), Error> {
        let tmp = tmp_path_for(dest);
        let result = (|| -> Result<(), Error> {
            // std::fs::copy uses clonefile / copy_file_range with reflink where the
            // filesystem supports it, so on APFS/Btrfs/XFS-with-reflink this is O(1).
            std::fs::copy(source, &tmp)?;
            let mut f = OpenOptions::new().write(true).open(&tmp)?;
            f.seek(SeekFrom::Start(0))?;
            f.write_all(header)?;
            f.sync_all()?;
            drop(f);
            std::fs::rename(&tmp, dest)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        result
    }

    fn write_full_rewrite(&self, source: &Path, dest: &Path, header: &[u8]) -> Result<(), Error> {
        let tmp = tmp_path_for(dest);
        let result = (|| -> Result<(), Error> {
            let mut src = BufReader::new(File::open(source)?);
            src.seek(SeekFrom::Start(self.tensor_data_offset))?;

            let mut dst = BufWriter::new(File::create(&tmp)?);
            dst.write_all(header)?;
            std::io::copy(&mut src, &mut dst)?;

            let dst_file = dst.into_inner().map_err(|e| Error::Io(e.into_error()))?;
            dst_file.sync_all()?;
            drop(dst_file);

            std::fs::rename(&tmp, dest)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        result
    }
}

fn tmp_path_for(dest: &Path) -> PathBuf {
    let mut s: OsString = dest.as_os_str().to_owned();
    s.push(".tmp");
    PathBuf::from(s)
}
