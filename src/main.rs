use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};

use gguf_surgeon::{
    Diff, GgufArray, GgufFile, GgufValue, GgufValueType, Origin, SaveMode, SavePath, Schema,
    Severity, Violation, apply_patch, builtin_schema, is_reserved_key, parse_patch,
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

    /// Save policy. `auto` (default) picks the cheapest sufficient path. `rewrite`
    /// always does a full rewrite. `in-place` refuses any save that would force a
    /// tensor-data shift (so you never accidentally pay a multi-gigabyte copy).
    #[arg(long, global = true, value_enum, default_value_t = SaveMode::Auto)]
    save_mode: SaveMode,

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
    /// `value` may be a literal string, `@FILE` to read from a file (useful for
    /// multi-KB chat templates), or `@-` to read from stdin. `@@` escapes a
    /// literal leading `@`. Array values are not supported here yet.
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
    /// `value` accepts the same forms as `set`: literal, `@FILE`, `@-`, `@@literal`.
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
    /// Inspect the file for likely problems (missing common keys, out-of-range
    /// values). Runs format-level checks plus the built-in suggestion schema
    /// and the user `--schema` if provided. Read-only; nothing is written.
    Check {
        file: PathBuf,
        /// Skip the built-in suggestion schema; only run format checks plus any
        /// explicit `--schema`.
        #[arg(long)]
        no_default_schema: bool,
    },
    /// Open the file in an interactive TUI browser.
    Edit { file: PathBuf },
    /// Edit array-valued metadata entries (set/push/pop/insert/remove/len).
    Array(ArrayCmd),
}

#[derive(Args)]
struct ArrayCmd {
    #[command(subcommand)]
    op: ArrayOp,
}

#[derive(Subcommand)]
enum ArrayOp {
    /// Replace the element at INDEX of an array-valued KEY.
    Set {
        file: PathBuf,
        key: String,
        index: usize,
        value: String,
        #[arg(short = 'y', long)]
        yes: bool,
    },
    /// Append VALUE to the end of an array-valued KEY.
    Push {
        file: PathBuf,
        key: String,
        value: String,
        #[arg(short = 'y', long)]
        yes: bool,
    },
    /// Remove the last element of an array-valued KEY.
    Pop {
        file: PathBuf,
        key: String,
        #[arg(short = 'y', long)]
        yes: bool,
    },
    /// Insert VALUE at INDEX of an array-valued KEY (existing elements shift right).
    Insert {
        file: PathBuf,
        key: String,
        index: usize,
        value: String,
        #[arg(short = 'y', long)]
        yes: bool,
    },
    /// Remove the element at INDEX of an array-valued KEY (later elements shift left).
    Remove {
        file: PathBuf,
        key: String,
        index: usize,
        #[arg(short = 'y', long)]
        yes: bool,
    },
    /// Print the length of an array-valued KEY (read-only).
    Len { file: PathBuf, key: String },
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
        save_mode: cli.save_mode,
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
        Cmd::Check {
            file,
            no_default_schema,
        } => check(&file, env.schema, !no_default_schema)?,
        Cmd::Edit { file } => {
            gguf_surgeon::tui::run(&file, env.schema, env.force, env.save_mode)?
        }
        Cmd::Array(c) => array_dispatch(c.op, &env)?,
    }
    Ok(())
}

struct Env<'a> {
    schema: Option<&'a Schema>,
    force: bool,
    save_mode: SaveMode,
}

fn list(path: &Path, max_value_width: usize) -> Result<()> {
    let f = GgufFile::read(path)?;
    let visible: Vec<&(String, GgufValue)> = f
        .metadata
        .iter()
        .filter(|(k, _)| !is_reserved_key(k))
        .collect();

    println!("# {} (GGUF v{})", path.display(), f.version);
    println!(
        "# {} tensors, {} metadata entries",
        f.tensor_count,
        visible.len()
    );
    println!();

    let key_w = visible.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
    let type_w = visible
        .iter()
        .map(|(_, v)| v.ty().as_str().len())
        .max()
        .unwrap_or(0);

    for (key, value) in &visible {
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
    if is_reserved_key(key) {
        bail!("`{key}` is managed automatically by the editor and is not user-visible");
    }
    let f = GgufFile::read(path)?;
    let Some((_, value)) = f.metadata.iter().find(|(k, _)| k == key) else {
        bail!("key not found: {key}");
    };
    print_value(value, limit);
    Ok(())
}

fn print_value(value: &GgufValue, limit: Option<usize>) {
    use std::io::IsTerminal;
    let trailing_newline_for_scalar = io::stdout().is_terminal();
    let _ = write_value(
        &mut io::stdout(),
        value,
        limit,
        trailing_newline_for_scalar,
    );
}

/// Render `value` to `out`. For scalar values (everything except arrays),
/// `trailing_newline` controls whether a final `\n` is appended — set it to
/// `false` when stdout is redirected so `gguf get FILE key > template.j2`
/// round-trips cleanly through `gguf set FILE key @template.j2`. Array output
/// is multi-line by definition and ignores the flag.
fn write_value(
    out: &mut impl Write,
    value: &GgufValue,
    limit: Option<usize>,
    trailing_newline: bool,
) -> io::Result<()> {
    match value {
        GgufValue::Uint8(n) => write!(out, "{n}")?,
        GgufValue::Int8(n) => write!(out, "{n}")?,
        GgufValue::Uint16(n) => write!(out, "{n}")?,
        GgufValue::Int16(n) => write!(out, "{n}")?,
        GgufValue::Uint32(n) => write!(out, "{n}")?,
        GgufValue::Int32(n) => write!(out, "{n}")?,
        GgufValue::Uint64(n) => write!(out, "{n}")?,
        GgufValue::Int64(n) => write!(out, "{n}")?,
        GgufValue::Float32(x) => write!(out, "{x}")?,
        GgufValue::Float64(x) => write!(out, "{x}")?,
        GgufValue::Bool(b) => write!(out, "{b}")?,
        GgufValue::String(s) => write!(out, "{s}")?,
        GgufValue::Array(arr) => {
            let total = arr.elements.len();
            let n = limit.map_or(total, |l| l.min(total));
            writeln!(out, "[{}; {}]", arr.element_type.as_str(), total)?;
            for (i, e) in arr.elements.iter().take(n).enumerate() {
                write!(out, "  [{i:>5}] ")?;
                write_inline(out, e)?;
                writeln!(out)?;
            }
            if n < total {
                writeln!(out, "  ... ({} more)", total - n)?;
            }
            // Array output is multi-line; the trailing-newline flag is for
            // scalars only.
            return Ok(());
        }
    }
    if trailing_newline {
        writeln!(out)?;
    }
    Ok(())
}

fn write_inline(out: &mut impl Write, value: &GgufValue) -> io::Result<()> {
    match value {
        GgufValue::Uint8(n) => write!(out, "{n}"),
        GgufValue::Int8(n) => write!(out, "{n}"),
        GgufValue::Uint16(n) => write!(out, "{n}"),
        GgufValue::Int16(n) => write!(out, "{n}"),
        GgufValue::Uint32(n) => write!(out, "{n}"),
        GgufValue::Int32(n) => write!(out, "{n}"),
        GgufValue::Uint64(n) => write!(out, "{n}"),
        GgufValue::Int64(n) => write!(out, "{n}"),
        GgufValue::Float32(x) => write!(out, "{x}"),
        GgufValue::Float64(x) => write!(out, "{x}"),
        GgufValue::Bool(b) => write!(out, "{b}"),
        GgufValue::String(s) => write!(out, "{s:?}"),
        GgufValue::Array(arr) => write!(out, "[{}; {}]", arr.element_type.as_str(), arr.elements.len()),
    }
}

fn set(path: &Path, key: &str, raw_value: &str, yes: bool, env: &Env) -> Result<()> {
    if is_reserved_key(key) {
        bail!("`{key}` is managed automatically by the editor; cannot be edited");
    }
    let mut f = GgufFile::read(path)?;
    let pos = f
        .metadata
        .iter()
        .position(|(k, _)| k == key)
        .with_context(|| format!("key not found: {key}"))?;
    let ty = f.metadata[pos].1.ty();
    let raw_value = resolve_value_arg(raw_value)
        .with_context(|| format!("could not resolve value for key {key}"))?;
    let new_value = parse_value(&raw_value, ty)
        .with_context(|| format!("could not parse value for key {key}"))?;

    let before = f.metadata.clone();
    f.metadata[pos].1 = new_value;
    finalize(path, f, &before, yes, env)
}

fn add(path: &Path, key: &str, ty_name: &str, raw_value: &str, yes: bool, env: &Env) -> Result<()> {
    if is_reserved_key(key) {
        bail!("`{key}` is reserved by the editor; pick a different key name");
    }
    let mut f = GgufFile::read(path)?;
    if f.metadata.iter().any(|(k, _)| k == key) {
        bail!("key already exists: {key} (use `gguf set` to modify it)");
    }
    let ty = GgufValueType::parse_name(ty_name)
        .with_context(|| format!("unknown type: {ty_name}"))?;
    if matches!(ty, GgufValueType::Array) {
        bail!("adding array values is not supported via the CLI yet");
    }
    let raw_value = resolve_value_arg(raw_value)
        .with_context(|| format!("could not resolve value for key {key}"))?;
    let value = parse_value(&raw_value, ty)
        .with_context(|| format!("could not parse value as {ty_name}"))?;

    let before = f.metadata.clone();
    f.metadata.push((key.to_string(), value));
    finalize(path, f, &before, yes, env)
}

fn rm(path: &Path, key: &str, yes: bool, env: &Env) -> Result<()> {
    if is_reserved_key(key) {
        bail!("`{key}` is managed automatically by the editor; cannot be removed");
    }
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

    print_save_summary(path, &f, env)?;

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
    f.write(path, path, env.save_mode)?;
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
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for (k, v) in &diff.additions {
        let _ = write!(out, "+ {k}: ");
        let _ = write_inline(&mut out, v);
        let _ = writeln!(out);
    }
    for (k, v) in &diff.removals {
        let _ = write!(out, "- {k}: ");
        let _ = write_inline(&mut out, v);
        let _ = writeln!(out);
    }
    for (k, old, new) in &diff.changes {
        let _ = write!(out, "~ {k}: ");
        let _ = write_inline(&mut out, old);
        let _ = write!(out, " -> ");
        let _ = write_inline(&mut out, new);
        let _ = writeln!(out);
    }
}

fn print_save_summary(path: &Path, after: &GgufFile, env: &Env) -> Result<()> {
    let size = std::fs::metadata(path)?.len();
    let predicted = after.pick_save_path();
    let summary = match (env.save_mode, predicted) {
        (SaveMode::Rewrite, _) => format!(
            "full rewrite (forced by --save-mode=rewrite; will copy {size} bytes through a temp file)"
        ),
        (SaveMode::InPlace, SavePath::FullRewrite) => format!(
            "WILL BE REFUSED — --save-mode=in-place but edit needs a full rewrite of {size} bytes"
        ),
        (_, SavePath::HeaderOverwrite) => format!(
            "header overwrite ({} byte header; tensor data left in place via copy-on-write where supported)",
            after.tensor_data_offset
        ),
        (_, SavePath::FullRewrite) => format!(
            "full rewrite (will copy {size} bytes through a temp file)"
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

fn array_dispatch(op: ArrayOp, env: &Env) -> Result<()> {
    match op {
        ArrayOp::Set {
            file,
            key,
            index,
            value,
            yes,
        } => array_mutate(&file, &key, env, yes, |arr| {
            check_index(arr, index)?;
            let new_elem = parse_value(&value, arr.element_type)
                .with_context(|| format!("could not parse value as {}", arr.element_type.as_str()))?;
            arr.elements[index] = new_elem;
            Ok(())
        }),
        ArrayOp::Push {
            file,
            key,
            value,
            yes,
        } => array_mutate(&file, &key, env, yes, |arr| {
            let new_elem = parse_value(&value, arr.element_type)
                .with_context(|| format!("could not parse value as {}", arr.element_type.as_str()))?;
            arr.elements.push(new_elem);
            Ok(())
        }),
        ArrayOp::Pop { file, key, yes } => array_mutate(&file, &key, env, yes, |arr| {
            if arr.elements.pop().is_none() {
                bail!("array is already empty");
            }
            Ok(())
        }),
        ArrayOp::Insert {
            file,
            key,
            index,
            value,
            yes,
        } => array_mutate(&file, &key, env, yes, |arr| {
            if index > arr.elements.len() {
                bail!(
                    "index {index} out of bounds for insert (length is {})",
                    arr.elements.len()
                );
            }
            let new_elem = parse_value(&value, arr.element_type)
                .with_context(|| format!("could not parse value as {}", arr.element_type.as_str()))?;
            arr.elements.insert(index, new_elem);
            Ok(())
        }),
        ArrayOp::Remove {
            file,
            key,
            index,
            yes,
        } => array_mutate(&file, &key, env, yes, |arr| {
            check_index(arr, index)?;
            arr.elements.remove(index);
            Ok(())
        }),
        ArrayOp::Len { file, key } => {
            let f = GgufFile::read(&file)?;
            let arr = find_array(&f, &key)?;
            println!("{}", arr.elements.len());
            Ok(())
        }
    }
}

fn check_index(arr: &GgufArray, index: usize) -> Result<()> {
    if index >= arr.elements.len() {
        bail!(
            "index {index} out of bounds for array of length {}",
            arr.elements.len()
        );
    }
    Ok(())
}

fn find_array<'a>(f: &'a GgufFile, key: &str) -> Result<&'a GgufArray> {
    let entry = f
        .metadata
        .iter()
        .find(|(k, _)| k == key)
        .with_context(|| format!("key not found: {key}"))?;
    match &entry.1 {
        GgufValue::Array(a) => Ok(a),
        other => bail!(
            "key {key} is not an array (it is a {})",
            other.ty().as_str()
        ),
    }
}

fn array_mutate(
    path: &Path,
    key: &str,
    env: &Env,
    yes: bool,
    op: impl FnOnce(&mut GgufArray) -> Result<()>,
) -> Result<()> {
    if is_reserved_key(key) {
        bail!("`{key}` is managed automatically by the editor; cannot be edited");
    }
    let mut f = GgufFile::read(path)?;
    let pos = f
        .metadata
        .iter()
        .position(|(k, _)| k == key)
        .with_context(|| format!("key not found: {key}"))?;
    let before = f.metadata.clone();
    let GgufValue::Array(ref mut arr) = f.metadata[pos].1 else {
        bail!(
            "key {key} is not an array (it is a {})",
            f.metadata[pos].1.ty().as_str()
        );
    };
    op(arr)?;
    finalize(path, f, &before, yes, env)
}

fn check(path: &Path, user_schema: Option<&Schema>, use_default: bool) -> Result<()> {
    let f = GgufFile::read(path)?;

    let mut violations = f.validate_format();
    if use_default {
        let default = builtin_schema();
        if default.applies_to_version(f.version) {
            violations.extend(default.validate(&f.metadata));
        }
    }
    if let Some(s) = user_schema.filter(|s| s.applies_to_version(f.version)) {
        violations.extend(s.validate(&f.metadata));
    }

    if violations.is_empty() {
        println!("{}: clean", path.display());
        return Ok(());
    }

    print_violations(&violations);
    let errors = violations
        .iter()
        .filter(|v| v.severity == Severity::Error)
        .count();
    let warnings = violations.len() - errors;
    println!();
    println!("{errors} error(s), {warnings} warning(s)");
    if errors > 0 {
        std::process::exit(1);
    }
    Ok(())
}

fn info(path: &Path) -> Result<()> {
    let f = GgufFile::read(path)?;
    let file_size = std::fs::metadata(path)?.len();
    let tensor_data_size = file_size.saturating_sub(f.tensor_data_offset);
    let header_padding = f.tensor_data_offset.saturating_sub(f.header_end);

    println!("file:                {}", path.display());
    println!("size (bytes):        {file_size}");
    println!("version:             {}", f.version);
    println!(
        "endianness:          {}",
        if f.little_endian { "little" } else { "big" }
    );
    println!("tensors:             {}", f.tensors.len());
    println!("metadata entries:    {}", f.metadata.len());
    println!("alignment:           {}", f.alignment);
    println!("header end offset:   {}", f.header_end);
    println!("tensor data offset:  {}", f.tensor_data_offset);
    println!("tensor data size:    {tensor_data_size}");
    println!("header padding:      {header_padding}");
    Ok(())
}

/// Resolve a CLI value argument: literal, `@FILE` (read file contents), `@-`
/// (read stdin), or `@@literal` (escaped literal `@literal`). The file form is
/// what makes pasting a multi-KB chat template ergonomic from the CLI; the
/// stdin form lets generators pipe values straight in. File contents are used
/// as-is — no trailing-newline stripping. If you need round-tripping with
/// `gguf get` (which prints a trailing newline), preprocess the file accordingly.
fn resolve_value_arg(raw: &str) -> Result<String> {
    let Some(rest) = raw.strip_prefix('@') else {
        return Ok(raw.to_string());
    };
    if let Some(literal) = rest.strip_prefix('@') {
        // `@@x` is a literal `@x` (escape).
        return Ok(format!("@{literal}"));
    }
    if rest == "-" {
        let mut s = String::new();
        io::stdin()
            .read_to_string(&mut s)
            .context("could not read value from stdin")?;
        return Ok(s);
    }
    std::fs::read_to_string(rest).with_context(|| format!("could not read value from {rest}"))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_value_arg_literal_passes_through() {
        assert_eq!(resolve_value_arg("hello").unwrap(), "hello");
        assert_eq!(resolve_value_arg("").unwrap(), "");
        assert_eq!(resolve_value_arg("with spaces and {jinja}").unwrap(),
                   "with spaces and {jinja}");
    }

    #[test]
    fn resolve_value_arg_double_at_escapes_literal() {
        assert_eq!(resolve_value_arg("@@hello").unwrap(), "@hello");
        assert_eq!(resolve_value_arg("@@").unwrap(), "@");
    }

    #[test]
    fn resolve_value_arg_at_file_reads_file() {
        let pid = std::process::id();
        let path = std::env::temp_dir().join(format!("ggufsurgeon-test-resolve-{pid}.txt"));
        let contents = "{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] }}{% endif %}";
        std::fs::write(&path, contents).unwrap();

        let arg = format!("@{}", path.display());
        let resolved = resolve_value_arg(&arg).unwrap();
        assert_eq!(resolved, contents);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn resolve_value_arg_missing_file_errors() {
        let arg = "@/this/path/does/not/exist/ggufsurgeon-bogus";
        assert!(resolve_value_arg(arg).is_err());
    }

    // Bug #1 regression test (round-trip newline drift). Exercises both halves
    // of the round-trip through the real code: `write_value` with
    // `trailing_newline = false` (the redirected-stdout mode) writes the bytes
    // that `gguf get FILE key > template.j2` produces, and `resolve_value_arg`
    // reads them back the way `gguf set FILE key @template.j2` does. The
    // round-trip is clean only when `write_value` suppresses the trailing
    // newline on non-TTY stdout — which is what the TTY-aware `print_value`
    // wrapper now does.
    #[test]
    fn round_trip_via_get_redirect_and_at_file_is_clean() {
        let pid = std::process::id();
        let path = std::env::temp_dir().join(format!("ggufsurgeon-roundtrip-bug-{pid}.txt"));

        let original = "the quick brown fox";
        let value = GgufValue::String(original.to_string());

        // Simulate `gguf get FILE key > path` (non-TTY): no trailing newline.
        let mut buf: Vec<u8> = Vec::new();
        write_value(&mut buf, &value, None, /*trailing_newline=*/ false).unwrap();
        std::fs::write(&path, &buf).unwrap();

        // Simulate `gguf set FILE key @path`.
        let arg = format!("@{}", path.display());
        let resolved = resolve_value_arg(&arg).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(
            resolved, original,
            "round-trip via `gguf get > file` + `gguf set @file` is not clean: \
             got {} bytes ({:?}), expected {} bytes ({:?})",
            resolved.len(), resolved, original.len(), original,
        );
    }
}
