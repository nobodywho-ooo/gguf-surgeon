use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};

use gguf_surgeon::{
    Diff, GgufFile, GgufValue, GgufValueType, Origin, SavePath, Schema, Severity, Violation,
    apply_patch, parse_patch,
};

#[derive(Parser)]
#[command(name = "gguf", about = "Editor for GGUF model file metadata.")]
struct Cli {
    /// Schema overlay (JSON) to validate against. Errors block save unless --force.
    #[arg(long, global = true, value_name = "PATH")]
    schema: Option<PathBuf>,

    /// Override schema-level errors and save anyway. Format-level errors are still unconditional.
    #[arg(long, global = true)]
    force: bool,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// List all metadata keys, types, and a value summary.
    List {
        file: PathBuf,
        /// Maximum characters to print per string value before truncating.
        #[arg(long, default_value_t = 80)]
        max_value_width: usize,
    },
    /// Print the value of a single metadata key, fully expanded.
    Get {
        file: PathBuf,
        key: String,
        /// Maximum number of array elements to print (default: all).
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Set a scalar metadata value, parsing the input as the existing key's type.
    /// In-place edit via temp + atomic rename. Array values are not supported here yet.
    Set {
        file: PathBuf,
        key: String,
        value: String,
        /// Skip the preview/confirmation prompt.
        #[arg(short = 'y', long)]
        yes: bool,
    },
    /// Add a new metadata entry. Fails if the key already exists.
    /// TYPE is one of: u8 i8 u16 i16 u32 i32 u64 i64 f32 f64 bool string.
    Add {
        file: PathBuf,
        key: String,
        #[arg(value_name = "TYPE")]
        ty: String,
        value: String,
        #[arg(short = 'y', long)]
        yes: bool,
    },
    /// Remove a metadata entry by key. Fails if the key does not exist.
    Rm {
        file: PathBuf,
        key: String,
        #[arg(short = 'y', long)]
        yes: bool,
    },
    /// Apply a JSON patch (an array of {op, key, ...} objects) to a file in one shot.
    /// Operations: set (key, value), add (key, type, value), rm (key).
    Patch {
        file: PathBuf,
        /// Path to the JSON patch file. Use `-` to read from stdin.
        patch_file: PathBuf,
        #[arg(short = 'y', long)]
        yes: bool,
    },
    /// Print a header summary: version, sizes, alignment, offsets, padding.
    Info { file: PathBuf },
    /// Open the file in an interactive TUI browser.
    Edit { file: PathBuf },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let schema = cli
        .schema
        .as_ref()
        .map(|p| Schema::load(p))
        .transpose()?;
    let env = Env {
        schema: schema.as_ref(),
        force: cli.force,
    };
    match cli.cmd {
        Cmd::List {
            file,
            max_value_width,
        } => list(&file, max_value_width)?,
        Cmd::Get { file, key, limit } => get(&file, &key, limit)?,
        Cmd::Set {
            file,
            key,
            value,
            yes,
        } => set(&file, &key, &value, yes, &env)?,
        Cmd::Add {
            file,
            key,
            ty,
            value,
            yes,
        } => add(&file, &key, &ty, &value, yes, &env)?,
        Cmd::Rm { file, key, yes } => rm(&file, &key, yes, &env)?,
        Cmd::Patch {
            file,
            patch_file,
            yes,
        } => patch(&file, &patch_file, yes, &env)?,
        Cmd::Info { file } => info(&file)?,
        Cmd::Edit { file } => gguf_surgeon::tui::run(&file)?,
    }
    Ok(())
}

struct Env<'a> {
    schema: Option<&'a Schema>,
    force: bool,
}

fn list(path: &Path, max_value_width: usize) -> Result<()> {
    let f = GgufFile::read(path)?;
    println!("# {} (GGUF v{})", path.display(), f.version);
    println!(
        "# {} tensors, {} metadata entries",
        f.tensor_count,
        f.metadata.len()
    );
    println!();

    let key_w = f.metadata.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
    let type_w = f
        .metadata
        .iter()
        .map(|(_, v)| v.ty().as_str().len())
        .max()
        .unwrap_or(0);

    for (key, value) in &f.metadata {
        let summary = summarize(value, max_value_width);
        println!(
            "{key:<key_w$}  {ty:<type_w$}  {summary}",
            key = key,
            ty = value.ty().as_str(),
            summary = summary,
        );
    }
    Ok(())
}

fn summarize(value: &GgufValue, max_width: usize) -> String {
    match value {
        GgufValue::Uint8(n) => n.to_string(),
        GgufValue::Int8(n) => n.to_string(),
        GgufValue::Uint16(n) => n.to_string(),
        GgufValue::Int16(n) => n.to_string(),
        GgufValue::Uint32(n) => n.to_string(),
        GgufValue::Int32(n) => n.to_string(),
        GgufValue::Float32(x) => format!("{x}"),
        GgufValue::Bool(b) => b.to_string(),
        GgufValue::String(s) => format_string(s, max_width),
        GgufValue::Array(arr) => format!("[{}; {}]", arr.element_type.as_str(), arr.elements.len()),
        GgufValue::Uint64(n) => n.to_string(),
        GgufValue::Int64(n) => n.to_string(),
        GgufValue::Float64(x) => format!("{x}"),
    }
}

fn format_string(s: &str, max_width: usize) -> String {
    let count = s.chars().count();
    if count <= max_width {
        format!("{s:?}")
    } else {
        let head: String = s.chars().take(max_width).collect();
        format!("{head:?}\u{2026} ({count} chars total)")
    }
}

fn get(path: &Path, key: &str, limit: Option<usize>) -> Result<()> {
    let f = GgufFile::read(path)?;
    let Some((_, value)) = f.metadata.iter().find(|(k, _)| k == key) else {
        bail!("key not found: {key}");
    };
    print_value(value, limit);
    Ok(())
}

fn print_value(value: &GgufValue, limit: Option<usize>) {
    match value {
        GgufValue::Uint8(n) => println!("{n}"),
        GgufValue::Int8(n) => println!("{n}"),
        GgufValue::Uint16(n) => println!("{n}"),
        GgufValue::Int16(n) => println!("{n}"),
        GgufValue::Uint32(n) => println!("{n}"),
        GgufValue::Int32(n) => println!("{n}"),
        GgufValue::Uint64(n) => println!("{n}"),
        GgufValue::Int64(n) => println!("{n}"),
        GgufValue::Float32(x) => println!("{x}"),
        GgufValue::Float64(x) => println!("{x}"),
        GgufValue::Bool(b) => println!("{b}"),
        GgufValue::String(s) => println!("{s}"),
        GgufValue::Array(arr) => {
            let total = arr.elements.len();
            let n = limit.map_or(total, |l| l.min(total));
            println!("[{}; {}]", arr.element_type.as_str(), total);
            for (i, e) in arr.elements.iter().take(n).enumerate() {
                print!("  [{i:>5}] ");
                print_inline(e);
                println!();
            }
            if n < total {
                println!("  ... ({} more)", total - n);
            }
        }
    }
}

fn print_inline(value: &GgufValue) {
    match value {
        GgufValue::Uint8(n) => print!("{n}"),
        GgufValue::Int8(n) => print!("{n}"),
        GgufValue::Uint16(n) => print!("{n}"),
        GgufValue::Int16(n) => print!("{n}"),
        GgufValue::Uint32(n) => print!("{n}"),
        GgufValue::Int32(n) => print!("{n}"),
        GgufValue::Uint64(n) => print!("{n}"),
        GgufValue::Int64(n) => print!("{n}"),
        GgufValue::Float32(x) => print!("{x}"),
        GgufValue::Float64(x) => print!("{x}"),
        GgufValue::Bool(b) => print!("{b}"),
        GgufValue::String(s) => print!("{s:?}"),
        GgufValue::Array(arr) => print!("[{}; {}]", arr.element_type.as_str(), arr.elements.len()),
    }
}

fn set(path: &Path, key: &str, raw_value: &str, yes: bool, env: &Env) -> Result<()> {
    let mut f = GgufFile::read(path)?;
    let pos = f
        .metadata
        .iter()
        .position(|(k, _)| k == key)
        .with_context(|| format!("key not found: {key}"))?;
    let ty = f.metadata[pos].1.ty();
    let new_value =
        parse_value(raw_value, ty).with_context(|| format!("could not parse value for key {key}"))?;

    let before = f.metadata.clone();
    f.metadata[pos].1 = new_value;
    finalize(path, f, &before, yes, env)
}

fn add(path: &Path, key: &str, ty_name: &str, raw_value: &str, yes: bool, env: &Env) -> Result<()> {
    let mut f = GgufFile::read(path)?;
    if f.metadata.iter().any(|(k, _)| k == key) {
        bail!("key already exists: {key} (use `gguf set` to modify it)");
    }
    let ty = GgufValueType::parse_name(ty_name)
        .with_context(|| format!("unknown type: {ty_name}"))?;
    if matches!(ty, GgufValueType::Array) {
        bail!("adding array values is not supported via the CLI yet");
    }
    let value = parse_value(raw_value, ty)
        .with_context(|| format!("could not parse value as {ty_name}"))?;

    let before = f.metadata.clone();
    f.metadata.push((key.to_string(), value));
    finalize(path, f, &before, yes, env)
}

fn rm(path: &Path, key: &str, yes: bool, env: &Env) -> Result<()> {
    let mut f = GgufFile::read(path)?;
    let pos = f
        .metadata
        .iter()
        .position(|(k, _)| k == key)
        .with_context(|| format!("key not found: {key}"))?;

    let before = f.metadata.clone();
    f.metadata.remove(pos);
    finalize(path, f, &before, yes, env)
}

fn patch(path: &Path, patch_path: &Path, yes: bool, env: &Env) -> Result<()> {
    let json = if patch_path == Path::new("-") {
        let mut s = String::new();
        io::stdin().read_to_string(&mut s)?;
        s
    } else {
        std::fs::read_to_string(patch_path)
            .with_context(|| format!("could not read patch file: {}", patch_path.display()))?
    };
    let patch = parse_patch(&json)?;

    let mut f = GgufFile::read(path)?;
    let before = f.metadata.clone();
    apply_patch(&mut f, &patch)?;
    finalize(path, f, &before, yes, env)
}

fn finalize(
    path: &Path,
    mut f: GgufFile,
    before: &[(String, GgufValue)],
    yes: bool,
    env: &Env,
) -> Result<()> {
    let diff = Diff::between(before, &f.metadata);
    print_diff(&diff);

    let mut violations = f.validate_format();
    if let Some(schema) = env.schema.filter(|s| s.applies_to_version(f.version)) {
        violations.extend(schema.validate(&f.metadata));
    }
    if !violations.is_empty() {
        print_violations(&violations);
    }

    print_save_summary(path, &f)?;

    let format_errors = violations
        .iter()
        .filter(|v| v.origin == Origin::Format && v.severity == Severity::Error)
        .count();
    if format_errors > 0 {
        bail!("save blocked: {format_errors} format error(s) (cannot be overridden — file would not load)");
    }
    let schema_errors = violations
        .iter()
        .filter(|v| v.origin == Origin::Schema && v.severity == Severity::Error)
        .count();
    if schema_errors > 0 && !env.force {
        bail!("save blocked: {schema_errors} schema error(s); pass --force to override");
    }

    if !yes && !confirm_prompt()? {
        eprintln!("aborted, no changes written");
        return Ok(());
    }
    f.write(path, path)?;
    Ok(())
}

fn print_violations(violations: &[Violation]) {
    for v in violations {
        let tag = match (v.origin, v.severity) {
            (Origin::Format, Severity::Error) => "format-err ",
            (Origin::Format, Severity::Warning) => "format-warn",
            (Origin::Schema, Severity::Error) => "schema-err ",
            (Origin::Schema, Severity::Warning) => "schema-warn",
        };
        println!("[{tag}] {}: {}", v.key, v.message);
    }
}

fn print_diff(diff: &Diff) {
    if diff.is_empty() {
        println!("(no changes)");
        return;
    }
    for (k, v) in &diff.additions {
        print!("+ {k}: ");
        print_inline(v);
        println!();
    }
    for (k, v) in &diff.removals {
        print!("- {k}: ");
        print_inline(v);
        println!();
    }
    for (k, old, new) in &diff.changes {
        print!("~ {k}: ");
        print_inline(old);
        print!(" -> ");
        print_inline(new);
        println!();
    }
}

fn print_save_summary(path: &Path, after: &GgufFile) -> Result<()> {
    let size = std::fs::metadata(path)?.len();
    let summary = match after.pick_save_path() {
        SavePath::HeaderOverwrite => format!(
            "header overwrite ({} byte header; tensor data left in place via copy-on-write where supported)",
            after.tensor_data_offset
        ),
        SavePath::FullRewrite => format!(
            "full rewrite (will copy {} bytes through a temp file)",
            size
        ),
    };
    println!("save path: {summary}");
    Ok(())
}

fn confirm_prompt() -> Result<bool> {
    print!("\nApply these changes? [y/N] ");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(matches!(input.trim().to_lowercase().as_str(), "y" | "yes"))
}

fn info(path: &Path) -> Result<()> {
    let f = GgufFile::read(path)?;
    let file_size = std::fs::metadata(path)?.len();
    let tensor_data_size = file_size.saturating_sub(f.tensor_data_offset);
    let header_padding = f.tensor_data_offset.saturating_sub(f.header_end);

    println!("file:                {}", path.display());
    println!("size (bytes):        {file_size}");
    println!("version:             {}", f.version);
    println!("endianness:          little");
    println!("tensors:             {}", f.tensors.len());
    println!("metadata entries:    {}", f.metadata.len());
    println!("alignment:           {}", f.alignment);
    println!("header end offset:   {}", f.header_end);
    println!("tensor data offset:  {}", f.tensor_data_offset);
    println!("tensor data size:    {tensor_data_size}");
    println!("header padding:      {header_padding}");
    Ok(())
}

fn parse_value(input: &str, ty: GgufValueType) -> Result<GgufValue> {
    Ok(match ty {
        GgufValueType::Uint8 => GgufValue::Uint8(input.parse()?),
        GgufValueType::Int8 => GgufValue::Int8(input.parse()?),
        GgufValueType::Uint16 => GgufValue::Uint16(input.parse()?),
        GgufValueType::Int16 => GgufValue::Int16(input.parse()?),
        GgufValueType::Uint32 => GgufValue::Uint32(input.parse()?),
        GgufValueType::Int32 => GgufValue::Int32(input.parse()?),
        GgufValueType::Uint64 => GgufValue::Uint64(input.parse()?),
        GgufValueType::Int64 => GgufValue::Int64(input.parse()?),
        GgufValueType::Float32 => GgufValue::Float32(input.parse()?),
        GgufValueType::Float64 => GgufValue::Float64(input.parse()?),
        GgufValueType::Bool => GgufValue::Bool(input.parse()?),
        GgufValueType::String => GgufValue::String(input.to_string()),
        GgufValueType::Array => bail!("setting array values is not supported via the CLI yet"),
    })
}
