# gguf-surgeon

Surgical metadata edits for GGUF model files: browse the key/value metadata stored in a `.gguf` file and modify fields in place without touching the tensor data.

## Background

GGUF (GGML Universal Format) is the binary container used by `llama.cpp` and related projects to ship LLM weights. Every file starts with a header that describes the model: architecture, name, quantization, context length, tokenizer vocabulary, chat template, and arbitrary user-defined keys. Tensor data follows after the header.

The metadata block is what tools like `llama.cpp`, Ollama, LM Studio, and `transformers` read to decide how to load and run the model. Fixing a wrong field (e.g. a broken chat template, a misnamed architecture, an incorrect EOS token id) currently means re-quantizing the model or running ad-hoc Python scripts — even though the change is just a few bytes in the header.

## Goal

Build an editor that can:

1. **Explore** — open a `.gguf` file and list every metadata key, its type, and its value. Handle all GGUF value types (uint/int 8/16/32/64, float32/64, bool, string, array).
2. **Modify** — let the user change, add, or remove metadata entries and write the result back to a valid `.gguf` file. The tensor payload must remain byte-identical and the file must still load in `llama.cpp`.

## Implementation

The editor is implemented in **Rust**. The language fits the constraints directly:

- **Memory safety on hostile binary input.** Parsing untrusted GGUF files is exactly the kind of work where C/C++ accumulate CVEs and where Rust's bounds checks and ownership model pay off — directly serving the *Robust against malformed input* principle below.
- **Endian-explicit, low-level I/O.** Reading typed primitives in either byte order is straightforward and zero-cost, which is needed because GGUF supports both endiannesses.
- **Streaming and mmap without GC pauses.** `memmap2` for header-region reads, `std::io::BufReader`/`BufWriter` for streaming tensor copies — multi-gigabyte files never enter RAM.
- **Type-tag dispatch with exhaustive matching.** GGUF's value-type enum maps to a Rust `enum`; `match` is checked at compile time, so a new primitive type added later is a hard compiler error rather than a silent fall-through.
- **Cross-platform single binary.** Same codebase for Linux, macOS, and Windows; one static binary per platform makes distribution painless.

v1 ships two interaction modes from the same binary: a **CLI** (`gguf list`, `gguf get`, `gguf set`, `gguf patch`) for one-shot and scripted use, and a **TUI** built with `ratatui` (`gguf edit`) for interactive browsing and editing. GGUF files almost always live on remote ML servers reachable only over SSH, so a terminal-native tool fits the audience; a graphical desktop UI is out of scope for v1.

## Design principles

The editor must keep working when the GGUF spec changes — without a rewrite. That shapes the design:

- **Generic, not key-aware.** The parser treats metadata as a flat list of `(key, type, value)` triples. It does not know that `general.architecture` or `tokenizer.ggml.tokens` exist or what they mean — and it does not need to. Well-known keys, vendor-specific keys, and arbitrary user-defined keys all flow through the same code path. New well-known keys appear automatically with no code changes.
- **Self-describing input.** Every value carries its type tag in the file. Read the type from the file, never infer it from the key name. Read the format version, alignment, byte-order, and tensor-data offset from the header rather than assuming them — GGUF supports both little- and big-endian, and the parser must respect the flag instead of assuming the host's endianness.
- **Preserve the unknown.** If a future spec adds a new primitive type the editor does not understand, fields of that type are kept as opaque bytes — still listable and round-trip safe, just not editable until support is added.
- **Schema overlay is optional and external.** Any "this key should be a non-empty string", "this key is an enum of these values", or "render this as a chat template" knowledge lives in a swappable schema file (JSON/TOML), not in the code. Updating for a new spec version means dropping in a new schema file. Overlays can layer — built-in defaults plus user overrides — and each declares the GGUF version range it applies to. The overlay file format itself has a defined schema and is validated on load: a malformed overlay is rejected, never silently ignored.
- **Always verify version compatibility on open.** Every time a file is opened, read the version field from the header and check it against the set of versions the editor knows. Known version → proceed. Unknown version → fail loudly with the version number; read-only exploration may still be offered (the format is largely self-describing), but writes are refused unless the user explicitly overrides. Never silently assume a newer version behaves like an older one.
- **Validate before writing.** Every save runs constraint checks; the same checks run on open and live during editing so problems are surfaced early. Two layers, with different save-time behavior:
  1. *Format-level* (always enforced, derived from the spec itself) — declared type matches actual encoding, integer values fit their declared width, strings are valid UTF-8, every array element shares the declared element type, keys are unique, token-id fields stay within the tokens array bounds, declared counts match actual lengths. Failures **block the save unconditionally** — there is no `--force` override, because the result would not be a valid GGUF file and the failure would just move from save-time to load-time.
  2. *Schema-level* (enforced when a schema overlay is loaded) — enums, regex patterns, numeric ranges, required-together fields, cross-field consistency. Each rule is tagged `warning` or `error` in the overlay:
     - *warnings* show up in the preview/diff but do not block the save.
     - *errors* block the save by default, but are overridable with an explicit `--force` flag (or a "save anyway?" confirmation in interactive mode), since schemas can be wrong or out of date and the user may know something the overlay does not.

     A missing overlay means no schema checks, not a bypass of the format checks.
- **Safe, atomic writes with three save paths.** GGUF files are gigabytes; a partial write loses the model. Every save goes through `write to temporary file → fsync → atomic rename over the original` (with an optional `.bak` of the previous version) — even trivial edits, so a crash never leaves the file half-written. What gets written depends on the edit:
  1. *Same-size edit* — the new value is the same byte length as the old. Only a few bytes of the header region are re-emitted; tensor data is not read or written.
  2. *Size change absorbed by padding* — the metadata block grows or shrinks, but the delta fits within reserved header padding. The header region (metadata + tensor_info + padding) is re-emitted; tensor data stays at the same absolute file offset and is not read or written.
  3. *Full rewrite* — the size delta exceeds the padding budget. Tensor data must shift to a new offset. The whole file is streamed through: header re-emitted, tensor-info offsets recomputed, tensor data copied chunk-by-chunk.

  The editor reserves **64 KB of header padding** on every save (configurable) so that most realistic edits — chat-template changes, vocab tweaks, key additions — fall into path 2 and never copy tensor data. Path 3 only fires when the budget is exhausted.

  Which path is used is also controllable per invocation via a `--save-mode` parameter:
  - `auto` (default) — pick the smallest sufficient path (1, 2, or 3 as the edit dictates).
  - `rewrite` — always do path 3, even when an in-place edit would suffice. Useful for compacting padding, defragmenting after many edits, or producing a clean fresh file on every save.
  - `in-place` — allow only paths 1 and 2. If an edit would force a full rewrite, the save is refused with a clear error rather than silently copying multiple gigabytes. Useful when disk space is tight or fast saves are required.
- **Streaming I/O.** Tensor bytes are never fully loaded into memory. Only the header and the tensor-descriptor table are parsed; tensor data flows through in fixed-size chunks during a rewrite. The editor must work on multi-gigabyte files on a machine with modest RAM.
- **Robust against malformed input.** Treat every input file as untrusted. Cap declared array, string, and tensor sizes against the actual file length, bounds-check every offset, reject overflowing or circular values. The parser must fail cleanly — never crash, hang, or allocate unboundedly — on a corrupt or hostile file.

The result: a spec change at most requires updating an external schema file — never a code release. A file from a future, unknown version is never silently corrupted. And a file written by this editor will not violate its own format or any constraint the loaded schema declares.

## Scope

- Read and write any GGUF version the header advertises (subject to the compatibility check), using only the type information present in the file.
- Preserve tensor data, alignment, and padding exactly.
- Support editing scalar values, strings, and arrays (including the tokenizer vocabulary and merges).
- Treat custom and arbitrary keys as first-class. Any key name in any namespace can be listed, edited, added, or removed. Well-known keys get no special privileges; unknown keys get no special restrictions. The optional schema overlay only affects display and validation hints — its absence never blocks editing.
- Round-trip safety: re-saving an unedited file produces a byte-identical output, even for files containing keys or types the editor has never seen before.
- Preview before save: every save shows a diff (added, removed, and changed keys, with old → new values), names which save path will be taken (same-size in-place, header-region rewrite, or full file rewrite) along with an estimate of bytes that will be copied, and waits for confirmation.
- Undo and redo of edits within a session.
- Non-interactive mode: a CLI form that applies a JSON or TOML patch to a file in one shot — for CI pipelines, batch jobs, and scripted model surgery.
- Optional provenance stamps. The editor can write `general.last_edited_by` / `general.last_edited_at` keys to record who edited the file and when. Off by default, since these keys are not part of the spec and other tools may not expect them.

## Out of scope (for now)

- Editing tensor weights or quantization.
- Converting between GGUF versions.
- A full GUI — a usable CLI/TUI is enough for the first cut.

## Future work

- **Linux-native in-place metadata growth.** Today, when metadata grows beyond the reserved padding budget, the only option is a full rewrite (~2× peak disk usage during save, full read/write of the tensor data). On Linux with ext4 or XFS and a recent kernel, `fallocate(FALLOC_FL_INSERT_RANGE)` can shift the entire tensor-data region forward by a block-aligned delta as an O(1) filesystem-metadata operation — no tensor bytes are read or written, and disk usage barely changes. Adding this as a fourth save path would make unbounded metadata growth nearly free on supported systems. Constraints: Linux-only (no macOS/APFS, no Windows/NTFS support), 4 KB block-aligned only, not atomic on its own (needs a careful failure-recovery wrapper). Out of scope for the first cut; revisit once the basic editor is stable.

## References

- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- `llama.cpp` reference implementation: https://github.com/ggerganov/llama.cpp
