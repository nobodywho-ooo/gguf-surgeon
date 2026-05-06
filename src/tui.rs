use std::io;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use crossterm::{
    event::{
        self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEventKind,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::Line,
    widgets::{Block, Borders, Cell, Clear, Paragraph, Row, Table, TableState},
};

use crate::diff::Diff;
use crate::format::{GgufFile, is_reserved_key};
use crate::save::SaveMode;
use crate::schema::{Origin, Schema, Severity, Violation};
use crate::value::{GgufValue, GgufValueType};

const ARRAY_DETAIL_LIMIT: usize = 200;

pub fn run(
    path: &Path,
    schema: Option<&Schema>,
    force: bool,
    save_mode: SaveMode,
) -> Result<()> {
    let file = GgufFile::read(path)?;
    let file_size = std::fs::metadata(path)?.len();

    let mut app = App::new(
        file,
        path.to_path_buf(),
        file_size,
        schema.cloned(),
        force,
        save_mode,
    );

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableBracketedPaste)?;
    let backend = CrosstermBackend::new(stdout);
    let mut term = Terminal::new(backend)?;

    let result = (|| -> Result<()> {
        loop {
            term.draw(|f| draw(f, &mut app))?;
            if matches!(app.mode, Mode::Saving) {
                // Run the slow save synchronously; the screen above shows the saving overlay.
                match app.run_save() {
                    Ok(()) => {
                        app.mode = Mode::List;
                        app.status = Some("saved".into());
                    }
                    Err(e) => {
                        app.mode = Mode::List;
                        app.status = Some(format!("save failed: {e}"));
                    }
                }
                continue;
            }
            match event::read()? {
                Event::Key(k) if k.kind == KeyEventKind::Press => {
                    if app.handle_key(k.code)? {
                        break;
                    }
                }
                Event::Paste(text) => app.handle_paste(&text),
                _ => {}
            }
        }
        Ok(())
    })();

    disable_raw_mode()?;
    execute!(term.backend_mut(), DisableBracketedPaste, LeaveAlternateScreen)?;
    result
}

struct App {
    file: GgufFile,
    original_metadata: Vec<(String, GgufValue)>,
    path: PathBuf,
    file_size: u64,
    schema: Option<Schema>,
    force: bool,
    save_mode: SaveMode,
    visible: Vec<usize>,
    cursor: usize,
    list_state: TableState,
    mode: Mode,
    search_buf: String,
    status: Option<String>,
    /// Cursor state for array editing. Shared between `ArrayList` and `ArrayInput`
    /// so the cursor stays put while the user types into the input prompt.
    array_list_state: TableState,
}

enum Mode {
    List,
    Search,
    Detail(usize),
    Edit {
        idx: usize,
        buf: String,
        error: Option<String>,
    },
    /// Browsing the elements of an array-valued metadata key.
    ArrayList { parent_idx: usize },
    /// Editing a single element, pushing, or inserting. The action determines
    /// where the parsed value lands when the user hits Enter.
    ArrayInput {
        parent_idx: usize,
        action: ArrayAction,
        buf: String,
        error: Option<String>,
    },
    SaveConfirm,
    Saving,
    QuitConfirm,
}

#[derive(Debug, Clone, Copy)]
enum ArrayAction {
    /// Replace the element at the given index.
    Edit(usize),
    /// Append a new element to the end.
    Push,
    /// Insert a new element before the given index.
    Insert(usize),
}

impl App {
    fn new(
        file: GgufFile,
        path: PathBuf,
        file_size: u64,
        schema: Option<Schema>,
        force: bool,
        save_mode: SaveMode,
    ) -> Self {
        let original_metadata = file.metadata.clone();
        let visible: Vec<usize> = file
            .metadata
            .iter()
            .enumerate()
            .filter(|(_, (k, _))| !is_reserved_key(k))
            .map(|(i, _)| i)
            .collect();
        let mut list_state = TableState::default();
        if !visible.is_empty() {
            list_state.select(Some(0));
        }
        Self {
            file,
            original_metadata,
            path,
            file_size,
            schema,
            force,
            save_mode,
            visible,
            cursor: 0,
            list_state,
            mode: Mode::List,
            search_buf: String::new(),
            status: None,
            array_list_state: TableState::default(),
        }
    }

    /// Compute the violations the save flow needs to surface. Format-level
    /// violations are unconditional; schema-level violations are blocked unless
    /// `self.force` is set (warnings are reported but never blocking).
    fn save_violations(&self) -> Vec<Violation> {
        let mut v = self.file.validate_format();
        if let Some(s) = self.schema.as_ref().filter(|s| s.applies_to_version(self.file.version)) {
            v.extend(s.validate(&self.file.metadata));
        }
        v
    }

    fn dirty(&self) -> bool {
        !Diff::between(&self.original_metadata, &self.file.metadata).is_empty()
    }

    fn handle_key(&mut self, code: KeyCode) -> Result<bool> {
        match self.mode {
            Mode::List => self.handle_list(code),
            Mode::Search => Ok(self.handle_search(code)),
            Mode::Detail(_) => Ok(self.handle_detail(code)),
            Mode::Edit { .. } => Ok(self.handle_edit(code)),
            Mode::ArrayList { .. } => Ok(self.handle_array_list(code)),
            Mode::ArrayInput { .. } => Ok(self.handle_array_input(code)),
            Mode::SaveConfirm => self.handle_save_confirm(code),
            Mode::QuitConfirm => Ok(self.handle_quit_confirm(code)),
            // Mode::Saving runs synchronously in the main loop and never reads keys.
            Mode::Saving => Ok(false),
        }
    }

    fn handle_list(&mut self, code: KeyCode) -> Result<bool> {
        self.status = None;
        match code {
            KeyCode::Char('q') | KeyCode::Esc => {
                if self.dirty() {
                    self.mode = Mode::QuitConfirm;
                } else {
                    return Ok(true);
                }
            }
            KeyCode::Char('j') | KeyCode::Down => self.move_cursor(1),
            KeyCode::Char('k') | KeyCode::Up => self.move_cursor(-1),
            KeyCode::Char('g') | KeyCode::Home => self.set_cursor(0),
            KeyCode::Char('G') | KeyCode::End => {
                self.set_cursor(self.visible.len().saturating_sub(1));
            }
            KeyCode::PageDown => self.move_cursor(10),
            KeyCode::PageUp => self.move_cursor(-10),
            KeyCode::Char('/') => {
                self.mode = Mode::Search;
                self.search_buf.clear();
            }
            KeyCode::Enter => {
                if let Some(&idx) = self.visible.get(self.cursor) {
                    self.mode = Mode::Detail(idx);
                }
            }
            KeyCode::Char('e') => {
                if let Some(&idx) = self.visible.get(self.cursor) {
                    let (_, v) = &self.file.metadata[idx];
                    if matches!(v, GgufValue::Array(_)) {
                        self.array_list_state.select(Some(0));
                        self.mode = Mode::ArrayList { parent_idx: idx };
                    } else {
                        self.mode = Mode::Edit {
                            idx,
                            buf: render_for_edit(v),
                            error: None,
                        };
                    }
                }
            }
            KeyCode::Char('s') => {
                if !self.dirty() {
                    self.status = Some("no changes to save".into());
                } else {
                    self.mode = Mode::SaveConfirm;
                }
            }
            _ => {}
        }
        Ok(false)
    }

    fn handle_search(&mut self, code: KeyCode) -> bool {
        match code {
            KeyCode::Esc => {
                self.search_buf.clear();
                self.refilter();
                self.mode = Mode::List;
            }
            KeyCode::Enter => {
                self.refilter();
                self.mode = Mode::List;
            }
            KeyCode::Backspace => {
                self.search_buf.pop();
            }
            KeyCode::Char(c) => self.search_buf.push(c),
            _ => {}
        }
        false
    }

    fn handle_detail(&mut self, code: KeyCode) -> bool {
        if matches!(code, KeyCode::Esc | KeyCode::Char('q') | KeyCode::Backspace) {
            self.mode = Mode::List;
        }
        false
    }

    fn handle_edit(&mut self, code: KeyCode) -> bool {
        let Mode::Edit { idx, buf, error } = &mut self.mode else {
            return false;
        };
        match code {
            KeyCode::Esc => {
                self.mode = Mode::List;
            }
            KeyCode::Backspace => {
                buf.pop();
                *error = None;
            }
            KeyCode::Char(c) => {
                buf.push(c);
                *error = None;
            }
            KeyCode::Enter => {
                let target_idx = *idx;
                let text = buf.clone();
                let ty = self.file.metadata[target_idx].1.ty();
                match parse_value_for_type(&text, ty) {
                    Ok(v) => {
                        self.file.metadata[target_idx].1 = v;
                        self.mode = Mode::List;
                        self.status = Some("value updated (unsaved)".into());
                    }
                    Err(e) => {
                        *error = Some(e.to_string());
                    }
                }
            }
            _ => {}
        }
        false
    }

    fn handle_array_list(&mut self, code: KeyCode) -> bool {
        let Mode::ArrayList { parent_idx } = self.mode else {
            return false;
        };
        let arr_len = match &self.file.metadata[parent_idx].1 {
            GgufValue::Array(a) => a.elements.len(),
            _ => return false,
        };
        let cursor = self.array_list_state.selected().unwrap_or(0);
        match code {
            KeyCode::Esc | KeyCode::Char('q') => self.mode = Mode::List,
            KeyCode::Char('j') | KeyCode::Down => {
                if arr_len > 0 {
                    let new = (cursor + 1).min(arr_len - 1);
                    self.array_list_state.select(Some(new));
                }
            }
            KeyCode::Char('k') | KeyCode::Up => {
                self.array_list_state.select(Some(cursor.saturating_sub(1)));
            }
            KeyCode::Char('g') | KeyCode::Home => self.array_list_state.select(Some(0)),
            KeyCode::Char('G') | KeyCode::End => {
                if arr_len > 0 {
                    self.array_list_state.select(Some(arr_len - 1));
                }
            }
            KeyCode::PageDown => {
                if arr_len > 0 {
                    let new = (cursor + 10).min(arr_len - 1);
                    self.array_list_state.select(Some(new));
                }
            }
            KeyCode::PageUp => {
                self.array_list_state.select(Some(cursor.saturating_sub(10)));
            }
            KeyCode::Char('e') | KeyCode::Enter => {
                if let GgufValue::Array(a) = &self.file.metadata[parent_idx].1
                    && let Some(elem) = a.elements.get(cursor)
                {
                    self.mode = Mode::ArrayInput {
                        parent_idx,
                        action: ArrayAction::Edit(cursor),
                        buf: render_for_edit(elem),
                        error: None,
                    };
                }
            }
            KeyCode::Char('a') => {
                self.mode = Mode::ArrayInput {
                    parent_idx,
                    action: ArrayAction::Push,
                    buf: String::new(),
                    error: None,
                };
            }
            KeyCode::Char('i') => {
                self.mode = Mode::ArrayInput {
                    parent_idx,
                    action: ArrayAction::Insert(cursor),
                    buf: String::new(),
                    error: None,
                };
            }
            KeyCode::Char('d') => {
                if arr_len > 0
                    && let GgufValue::Array(a) = &mut self.file.metadata[parent_idx].1
                {
                    a.elements.remove(cursor);
                    let new_cursor = if a.elements.is_empty() {
                        0
                    } else {
                        cursor.min(a.elements.len() - 1)
                    };
                    self.array_list_state.select(Some(new_cursor));
                    self.status = Some("element removed (unsaved)".into());
                }
            }
            _ => {}
        }
        false
    }

    fn handle_array_input(&mut self, code: KeyCode) -> bool {
        let Mode::ArrayInput {
            parent_idx,
            action,
            ref mut buf,
            ref mut error,
        } = self.mode
        else {
            return false;
        };
        match code {
            KeyCode::Esc => {
                self.mode = Mode::ArrayList { parent_idx };
            }
            KeyCode::Backspace => {
                buf.pop();
                *error = None;
            }
            KeyCode::Char(c) => {
                buf.push(c);
                *error = None;
            }
            KeyCode::Enter => {
                let text = buf.clone();
                let elem_type = match &self.file.metadata[parent_idx].1 {
                    GgufValue::Array(a) => a.element_type,
                    _ => return false,
                };
                match parse_value_for_type(&text, elem_type) {
                    Ok(v) => {
                        let GgufValue::Array(ref mut a) = self.file.metadata[parent_idx].1
                        else {
                            return false;
                        };
                        let final_cursor = match action {
                            ArrayAction::Edit(idx) => {
                                if idx < a.elements.len() {
                                    a.elements[idx] = v;
                                }
                                idx
                            }
                            ArrayAction::Push => {
                                a.elements.push(v);
                                a.elements.len() - 1
                            }
                            ArrayAction::Insert(idx) => {
                                let pos = idx.min(a.elements.len());
                                a.elements.insert(pos, v);
                                pos
                            }
                        };
                        self.array_list_state.select(Some(final_cursor));
                        self.mode = Mode::ArrayList { parent_idx };
                        self.status = Some("element updated (unsaved)".into());
                    }
                    Err(e) => {
                        *error = Some(e.to_string());
                    }
                }
            }
            _ => {}
        }
        false
    }

    fn handle_save_confirm(&mut self, code: KeyCode) -> Result<bool> {
        match code {
            KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => {
                // Fast checks before transitioning into the saving overlay. The slow
                // tensor-data copy happens in run_save().
                let violations = self.save_violations();
                let format_errors = violations
                    .iter()
                    .filter(|v| v.origin == Origin::Format && v.severity == Severity::Error)
                    .count();
                if format_errors > 0 {
                    self.status = Some(format!(
                        "save blocked by {format_errors} format error(s)"
                    ));
                    self.mode = Mode::List;
                    return Ok(false);
                }
                let schema_errors = violations
                    .iter()
                    .filter(|v| v.origin == Origin::Schema && v.severity == Severity::Error)
                    .count();
                if schema_errors > 0 && !self.force {
                    self.status = Some(format!(
                        "save blocked by {schema_errors} schema error(s); pass --force on the CLI"
                    ));
                    self.mode = Mode::List;
                    return Ok(false);
                }
                self.mode = Mode::Saving;
            }
            KeyCode::Esc | KeyCode::Char('n') | KeyCode::Char('N') => {
                self.mode = Mode::List;
            }
            _ => {}
        }
        Ok(false)
    }

    fn handle_paste(&mut self, text: &str) {
        match &mut self.mode {
            Mode::Edit { buf, error, .. } | Mode::ArrayInput { buf, error, .. } => {
                buf.push_str(text);
                *error = None;
            }
            _ => {}
        }
    }

    fn run_save(&mut self) -> Result<()> {
        self.file.write(&self.path, &self.path, self.save_mode)?;
        self.file_size = std::fs::metadata(&self.path)?.len();
        self.file = GgufFile::read(&self.path)
            .map_err(|e| anyhow!("could not re-read after save: {e}"))?;
        self.original_metadata = self.file.metadata.clone();
        self.refilter();
        Ok(())
    }

    fn handle_quit_confirm(&mut self, code: KeyCode) -> bool {
        match code {
            KeyCode::Char('y') | KeyCode::Char('Y') => return true,
            KeyCode::Esc | KeyCode::Char('n') | KeyCode::Char('N') => {
                self.mode = Mode::List;
            }
            _ => {}
        }
        false
    }

    fn move_cursor(&mut self, delta: i32) {
        if self.visible.is_empty() {
            return;
        }
        let max = self.visible.len() as i32 - 1;
        let new = (self.cursor as i32 + delta).clamp(0, max) as usize;
        self.set_cursor(new);
    }

    fn set_cursor(&mut self, idx: usize) {
        let max = self.visible.len().saturating_sub(1);
        self.cursor = idx.min(max);
        self.list_state.select(Some(self.cursor));
    }

    fn refilter(&mut self) {
        let q = self.search_buf.to_lowercase();
        self.visible = self
            .file
            .metadata
            .iter()
            .enumerate()
            .filter(|(_, (k, _))| !is_reserved_key(k))
            .filter(|(_, (k, _))| q.is_empty() || k.to_lowercase().contains(&q))
            .map(|(i, _)| i)
            .collect();
        self.set_cursor(0);
    }
}

fn parse_value_for_type(input: &str, ty: GgufValueType) -> Result<GgufValue> {
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
        GgufValueType::Array => return Err(anyhow!("arrays not supported in TUI editor")),
    })
}

fn render_edit_buffer(buf: &str) -> String {
    if buf.contains('\n') {
        let chars = buf.chars().count();
        let lines = buf.split('\n').count();
        format!("<multiline: {chars} chars across {lines} lines>")
    } else if buf.chars().count() > 80 {
        let head: String = buf.chars().take(60).collect();
        format!("{head}\u{2026} <{} chars>", buf.chars().count())
    } else {
        buf.to_string()
    }
}

fn render_for_edit(v: &GgufValue) -> String {
    match v {
        GgufValue::String(s) => s.clone(),
        GgufValue::Bool(b) => b.to_string(),
        GgufValue::Uint8(n) => n.to_string(),
        GgufValue::Int8(n) => n.to_string(),
        GgufValue::Uint16(n) => n.to_string(),
        GgufValue::Int16(n) => n.to_string(),
        GgufValue::Uint32(n) => n.to_string(),
        GgufValue::Int32(n) => n.to_string(),
        GgufValue::Uint64(n) => n.to_string(),
        GgufValue::Int64(n) => n.to_string(),
        GgufValue::Float32(x) => format!("{x}"),
        GgufValue::Float64(x) => format!("{x}"),
        GgufValue::Array(_) => String::new(),
    }
}

fn draw(f: &mut ratatui::Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(3),
        ])
        .split(f.area());

    draw_title(f, chunks[0], app);
    let array_parent = match &app.mode {
        Mode::ArrayList { parent_idx } | Mode::ArrayInput { parent_idx, .. } => Some(*parent_idx),
        _ => None,
    };
    if let Some(parent_idx) = array_parent {
        draw_body_array(f, chunks[1], app, parent_idx);
    } else {
        draw_body(f, chunks[1], app);
    }
    draw_status(f, chunks[2], app);

    match &app.mode {
        Mode::Detail(idx) => draw_detail(f, *idx, app),
        Mode::SaveConfirm => draw_save_confirm(f, app),
        Mode::Saving => draw_saving(f, app),
        Mode::QuitConfirm => draw_quit_confirm(f),
        _ => {}
    }
}

fn draw_title(f: &mut ratatui::Frame, area: Rect, app: &App) {
    let dirty_mark = if app.dirty() { " [unsaved]" } else { "" };
    let title = format!(
        " {} (v{}, {} bytes, {} tensors, {} metadata){} ",
        app.path.display(),
        app.file.version,
        app.file_size,
        app.file.tensors.len(),
        app.file.metadata.len(),
        dirty_mark,
    );
    let p = Paragraph::new(title).block(Block::default().borders(Borders::ALL));
    f.render_widget(p, area);
}

fn draw_body(f: &mut ratatui::Frame, area: Rect, app: &mut App) {
    let rows: Vec<Row> = app
        .visible
        .iter()
        .map(|&i| {
            let (k, v) = &app.file.metadata[i];
            Row::new(vec![
                Cell::from(k.as_str()),
                Cell::from(v.ty().as_str()),
                Cell::from(summarize(v, 80)),
            ])
        })
        .collect();
    let widths = [
        Constraint::Percentage(40),
        Constraint::Length(8),
        Constraint::Min(0),
    ];
    let table = Table::new(rows, widths)
        .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED));
    f.render_stateful_widget(table, area, &mut app.list_state);
}

fn draw_body_array(f: &mut ratatui::Frame, area: Rect, app: &mut App, parent_idx: usize) {
    let arr = match &app.file.metadata[parent_idx].1 {
        GgufValue::Array(a) => a,
        _ => return,
    };
    let rows: Vec<Row> = arr
        .elements
        .iter()
        .enumerate()
        .map(|(i, e)| {
            Row::new(vec![
                Cell::from(format!("[{i:>5}]")),
                Cell::from(summarize(e, 120)),
            ])
        })
        .collect();
    let widths = [Constraint::Length(9), Constraint::Min(0)];
    let table = Table::new(rows, widths)
        .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED));
    f.render_stateful_widget(table, area, &mut app.array_list_state);
}

fn draw_status(f: &mut ratatui::Frame, area: Rect, app: &App) {
    let text = match &app.mode {
        Mode::List => {
            let counter = if app.visible.is_empty() {
                "0 of 0".to_string()
            } else {
                format!("{} of {}", app.cursor + 1, app.visible.len())
            };
            let base = format!(
                "[{counter}]  [/] search  [enter] details  [e] edit  [s] save  [q] quit  [j/k] move",
            );
            if let Some(s) = &app.status {
                format!("{base}    {s}")
            } else {
                base
            }
        }
        Mode::Search => format!("search: {}_  [enter] apply  [esc] cancel", app.search_buf),
        Mode::Detail(_) => "[esc] back".to_string(),
        Mode::Edit { buf, error, .. } => {
            let display = render_edit_buffer(buf);
            let head = format!("edit: {display}_  [enter] apply  [esc] cancel");
            match error {
                Some(e) => format!("{head}    !{e}"),
                None => head,
            }
        }
        Mode::ArrayList { parent_idx } => {
            let key = &app.file.metadata[*parent_idx].0;
            let len = match &app.file.metadata[*parent_idx].1 {
                GgufValue::Array(a) => a.elements.len(),
                _ => 0,
            };
            let cursor = app.array_list_state.selected().unwrap_or(0);
            let pos = if len == 0 {
                "0 of 0".to_string()
            } else {
                format!("{} of {}", cursor + 1, len)
            };
            format!("array {key} [{pos}]  [e] edit  [a] append  [i] insert  [d] delete  [esc] back")
        }
        Mode::ArrayInput { action, buf, error, .. } => {
            let label = match action {
                ArrayAction::Edit(idx) => format!("edit [{idx}]"),
                ArrayAction::Push => "push".to_string(),
                ArrayAction::Insert(idx) => format!("insert before [{idx}]"),
            };
            let display = render_edit_buffer(buf);
            let head = format!("array {label}: {display}_  [enter] apply  [esc] cancel");
            match error {
                Some(e) => format!("{head}    !{e}"),
                None => head,
            }
        }
        Mode::SaveConfirm => "save? [y] yes  [n/esc] cancel".to_string(),
        Mode::Saving => "saving... do not close the terminal".to_string(),
        Mode::QuitConfirm => "discard unsaved changes? [y] yes  [n/esc] cancel".to_string(),
    };
    let p = Paragraph::new(text).block(Block::default().borders(Borders::ALL));
    f.render_widget(p, area);
}

fn draw_detail(f: &mut ratatui::Frame, idx: usize, app: &App) {
    let area = centered_rect(85, 85, f.area());
    let (k, v) = &app.file.metadata[idx];
    let title = format!(" {k} ({}) ", type_label(v));
    let lines = detail_lines(v);
    let p = Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(title));
    f.render_widget(Clear, area);
    f.render_widget(p, area);
}

fn draw_save_confirm(f: &mut ratatui::Frame, app: &App) {
    let area = centered_rect(70, 70, f.area());
    let diff = Diff::between(&app.original_metadata, &app.file.metadata);
    let mut lines: Vec<Line> = Vec::new();
    for (k, v) in &diff.additions {
        lines.push(Line::raw(format!("+ {k}: {}", inline_value(v))));
    }
    for (k, v) in &diff.removals {
        lines.push(Line::raw(format!("- {k}: {}", inline_value(v))));
    }
    for (k, old, new) in &diff.changes {
        lines.push(Line::raw(format!(
            "~ {k}: {} -> {}",
            inline_value(old),
            inline_value(new),
        )));
    }
    if lines.is_empty() {
        lines.push(Line::raw("(no changes)"));
    }

    let violations = app.save_violations();
    if !violations.is_empty() {
        lines.push(Line::raw(""));
        for v in &violations {
            let tag = match (v.origin, v.severity) {
                (Origin::Format, Severity::Error) => "format-err ",
                (Origin::Format, Severity::Warning) => "format-warn",
                (Origin::Schema, Severity::Error) => "schema-err ",
                (Origin::Schema, Severity::Warning) => "schema-warn",
            };
            lines.push(Line::raw(format!("[{tag}] {}: {}", v.key, v.message)));
        }
    }

    lines.push(Line::raw(""));
    let path = match app.file.predict_save_path(crate::format::DEFAULT_PADDING_STEP) {
        crate::save::SavePath::HeaderOverwrite => "header overwrite (CoW where supported)",
        crate::save::SavePath::FullRewrite => "full rewrite (will copy tensor data)",
    };
    lines.push(Line::raw(format!("save path: {path}")));

    let p = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" save? "));
    f.render_widget(Clear, area);
    f.render_widget(p, area);
}

fn draw_quit_confirm(f: &mut ratatui::Frame) {
    let area = centered_rect(50, 30, f.area());
    let p = Paragraph::new("There are unsaved changes.\nDiscard and quit?")
        .block(Block::default().borders(Borders::ALL).title(" quit? "));
    f.render_widget(Clear, area);
    f.render_widget(p, area);
}

fn draw_saving(f: &mut ratatui::Frame, app: &App) {
    let area = centered_rect(60, 30, f.area());
    let path = match app.file.predict_save_path(crate::format::DEFAULT_PADDING_STEP) {
        crate::save::SavePath::HeaderOverwrite => "header overwrite (fast)",
        crate::save::SavePath::FullRewrite => "full rewrite (copying tensor data)",
    };
    let lines = vec![
        Line::raw("Saving..."),
        Line::raw(""),
        Line::raw(format!("path: {path}")),
        Line::raw(""),
        Line::raw("Large files may take a while. Do not interrupt."),
    ];
    let p = Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" saving "));
    f.render_widget(Clear, area);
    f.render_widget(p, area);
}

fn type_label(v: &GgufValue) -> String {
    match v {
        GgufValue::Array(a) => format!("array<{}>", a.element_type.as_str()),
        _ => v.ty().as_str().to_string(),
    }
}

fn detail_lines(v: &GgufValue) -> Vec<Line<'static>> {
    match v {
        GgufValue::String(s) => vec![Line::raw(s.clone())],
        GgufValue::Array(a) => {
            let total = a.elements.len();
            let take = total.min(ARRAY_DETAIL_LIMIT);
            let mut lines: Vec<Line> = a
                .elements
                .iter()
                .take(take)
                .enumerate()
                .map(|(i, e)| Line::raw(format!("[{i:>5}] {}", inline_value(e))))
                .collect();
            if take < total {
                lines.push(Line::raw(format!("...{} more", total - take)));
            }
            lines
        }
        _ => vec![Line::raw(inline_value(v))],
    }
}

fn inline_value(v: &GgufValue) -> String {
    match v {
        GgufValue::Uint8(n) => n.to_string(),
        GgufValue::Int8(n) => n.to_string(),
        GgufValue::Uint16(n) => n.to_string(),
        GgufValue::Int16(n) => n.to_string(),
        GgufValue::Uint32(n) => n.to_string(),
        GgufValue::Int32(n) => n.to_string(),
        GgufValue::Uint64(n) => n.to_string(),
        GgufValue::Int64(n) => n.to_string(),
        GgufValue::Float32(x) => format!("{x}"),
        GgufValue::Float64(x) => format!("{x}"),
        GgufValue::Bool(b) => b.to_string(),
        GgufValue::String(s) => format!("{s:?}"),
        GgufValue::Array(a) => format!("[{}; {}]", a.element_type.as_str(), a.elements.len()),
    }
}

fn summarize(v: &GgufValue, max: usize) -> String {
    match v {
        GgufValue::String(s) => {
            let count = s.chars().count();
            if count <= max {
                format!("{s:?}")
            } else {
                let head: String = s.chars().take(max).collect();
                format!("{head:?}\u{2026} ({count} chars)")
            }
        }
        GgufValue::Array(a) => format!("[{}; {}]", a.element_type.as_str(), a.elements.len()),
        _ => inline_value(v),
    }
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let v = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(v[1])[1]
}
