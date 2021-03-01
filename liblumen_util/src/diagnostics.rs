use std::io::Write;
use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub type DisplayConfig = libeir_diagnostics::term::Config;
pub type DisplayStyle = libeir_diagnostics::term::DisplayStyle;
pub type DisplayChars = libeir_diagnostics::term::Chars;

pub use libeir_diagnostics::term::termcolor::*;
pub use libeir_diagnostics::term::{ColorArg, Styles};
pub use libeir_diagnostics::{
    ByteIndex, CodeMap, FileName, Files, SourceFile, SourceId, SourceIndex, SourceSpan,
};
pub use libeir_diagnostics::{Diagnostic, Label, LabelStyle, Severity};

use crate::error::{FatalError, Verbosity};

#[derive(Debug, Clone)]
pub struct DiagnosticsConfig {
    pub warnings_as_errors: bool,
    pub no_warn: bool,
    pub display: DisplayConfig,
}

pub trait Emitter {
    fn buffer(&self) -> Buffer;
    fn print(&self, buffer: &Buffer) -> std::io::Result<()>;
}

pub struct DefaultEmitter {
    writer: BufferWriter,
}
impl DefaultEmitter {
    pub fn new(color: ColorChoice) -> Self {
        let writer = BufferWriter::stderr(color);
        Self { writer }
    }
}
impl Emitter for DefaultEmitter {
    #[inline(always)]
    fn buffer(&self) -> Buffer {
        self.writer.buffer()
    }

    #[inline(always)]
    fn print(&self, buffer: &Buffer) -> std::io::Result<()> {
        self.writer.print(buffer)
    }
}

#[derive(Clone, Copy, Default)]
pub struct NullEmitter {
    ansi: bool,
}
impl NullEmitter {
    pub fn new(color: ColorChoice) -> Self {
        let ansi = match color {
            ColorChoice::Never => false,
            ColorChoice::Always | ColorChoice::AlwaysAnsi => true,
            ColorChoice::Auto => {
                if atty::is(atty::Stream::Stdout) {
                    true
                } else {
                    false
                }
            }
        };
        Self { ansi }
    }
}
impl Emitter for NullEmitter {
    #[inline(always)]
    fn buffer(&self) -> Buffer {
        if self.ansi {
            Buffer::ansi()
        } else {
            Buffer::no_color()
        }
    }

    #[inline(always)]
    fn print(&self, _buffer: &Buffer) -> std::io::Result<()> {
        Ok(())
    }
}

/// Construct an in-flight diagnostic
pub struct InFlightDiagnostic<'h> {
    handler: &'h DiagnosticsHandler,
    file_id: Option<SourceId>,
    #[allow(dead_code)]
    filename: Option<FileName>,
    diagnostic: Diagnostic,
    severity: Severity,
}
impl<'h> InFlightDiagnostic<'h> {
    fn new(handler: &'h DiagnosticsHandler, severity: Severity) -> Self {
        Self {
            handler,
            file_id: None,
            filename: None,
            diagnostic: Diagnostic::new(severity),
            severity,
        }
    }

    /// Returns the severity level of this diagnostic
    pub fn severity(&self) -> Severity {
        self.severity
    }

    /// Returns whether this diagnostic should be generated
    /// with verbose detail. Intended to be used when building
    /// diagnostics in-flight by formatting functions which do
    /// not know what the current diagnostic configuration is
    pub fn verbose(&self) -> bool {
        use libeir_diagnostics::term::DisplayStyle;
        match self.handler.display.display_style {
            DisplayStyle::Rich => true,
            _ => false,
        }
    }

    pub fn set_source_file(&mut self, filename: impl Into<FileName>) {
        let filename = filename.into();
        let file_id = self.handler.codemap.get_file_id(&filename);
        self.filename = Some(filename);
        self.file_id = file_id;
    }

    pub fn with_message(&mut self, message: impl Into<String>) {
        self.diagnostic.message = message.into();
    }

    pub fn with_primary_label(&mut self, line: u32, column: u32, message: Option<String>) {
        self.with_label_and_file_id(
            LabelStyle::Primary,
            self.file_id.clone(),
            line,
            column,
            message,
        )
    }

    pub fn with_secondary_label(&mut self, line: u32, column: u32, message: Option<String>) {
        self.with_label_and_file_id(
            LabelStyle::Secondary,
            self.file_id.clone(),
            line,
            column,
            message,
        )
    }

    pub fn with_label(
        &mut self,
        style: LabelStyle,
        filename: Option<FileName>,
        line: u32,
        column: u32,
        message: Option<String>,
    ) {
        if let Some(name) = filename {
            let id = self.handler.lookup_file_id(name);
            self.with_label_and_file_id(style, id, line, column, message);
        }
    }

    fn with_label_and_file_id(
        &mut self,
        style: LabelStyle,
        file_id: Option<SourceId>,
        line: u32,
        _column: u32,
        message: Option<String>,
    ) {
        if let Some(id) = file_id {
            let source_file = self.handler.codemap.get(id).unwrap();
            let line_index = (line - 1).into();
            let span = source_file
                .line_span(line_index)
                .expect("invalid line index");
            let label = if let Some(msg) = message {
                Label::new(style, id, span).with_message(msg)
            } else {
                Label::new(style, id, span)
            };
            self.diagnostic.labels.push(label);
        }
    }

    pub fn with_note(&mut self, note: impl Into<String>) {
        self.diagnostic.notes.push(note.into());
    }

    /// Emit the diagnostic via the DiagnosticHandler
    pub fn emit(self) {
        self.handler.emit(&self.diagnostic);
    }
}

pub struct DiagnosticsHandler {
    emitter: Arc<dyn Emitter>,
    codemap: Arc<CodeMap>,
    err_count: AtomicUsize,
    warnings_as_errors: bool,
    no_warn: bool,
    display: DisplayConfig,
}
// We can safely implement these traits for DiagnosticsHandler,
// as the only two non-atomic fields are read-only after creation
unsafe impl Send for DiagnosticsHandler {}
unsafe impl Sync for DiagnosticsHandler {}
impl DiagnosticsHandler {
    pub fn new(
        config: DiagnosticsConfig,
        codemap: Arc<CodeMap>,
        emitter: Arc<dyn Emitter>,
    ) -> Self {
        Self {
            emitter,
            codemap,
            err_count: AtomicUsize::new(0),
            warnings_as_errors: config.warnings_as_errors,
            no_warn: config.no_warn,
            display: config.display,
        }
    }

    pub fn lookup_file_id(&self, filename: impl Into<FileName>) -> Option<SourceId> {
        let filename = filename.into();
        self.codemap.get_file_id(&filename)
    }

    pub fn has_errors(&self) -> bool {
        self.err_count.load(Ordering::Relaxed) > 0
    }

    pub fn abort_if_errors(&self) {
        if self.has_errors() {
            FatalError.raise();
        }
    }

    pub fn fatal(&self, err: impl Into<String>) -> FatalError {
        self.error(err);
        FatalError
    }

    pub fn error(&self, err: impl Into<String>) {
        self.err_count.fetch_add(1, Ordering::Relaxed);
        let diagnostic = Diagnostic::error().with_message(err);
        self.emit(&diagnostic);
    }

    pub fn warn(&self, message: impl Into<String>) {
        if self.warnings_as_errors {
            self.error(message)
        } else if !self.no_warn {
            let diagnostic = Diagnostic::warning().with_message(message);
            self.emit(&diagnostic)
        }
    }

    pub fn info(&self, message: impl Into<String>) {
        let info_color = self.display.styles.header(Severity::Help);
        let mut buffer = self.emitter.buffer();
        buffer.set_color(&info_color).ok();
        write!(&mut buffer, "info").unwrap();
        buffer.set_color(&self.display.styles.header_message).ok();
        write!(&mut buffer, ": {}", message.into()).unwrap();
        buffer.reset().ok();
        write!(&mut buffer, "\n").unwrap();
        self.emitter.print(&buffer).unwrap();
    }

    pub fn debug(&self, message: impl Into<String>) {
        let mut debug_color = self.display.styles.header_message.clone();
        debug_color.set_fg(Some(Color::Blue));
        let mut buffer = self.emitter.buffer();
        buffer.set_color(&debug_color).ok();
        write!(&mut buffer, "debug").unwrap();
        buffer.set_color(&self.display.styles.header_message).ok();
        write!(&mut buffer, ": {}", message.into()).unwrap();
        buffer.reset().ok();
        write!(&mut buffer, "\n").unwrap();
        self.emitter.print(&buffer).unwrap();
    }

    pub fn note(&self, message: impl Into<String>) {
        let diagnostic = Diagnostic::note().with_message(message);
        self.emit(&diagnostic);
    }

    pub fn success(&self, prefix: &str, message: impl Into<String>) {
        self.write_prefixed(self.display.styles.header(Severity::Note), prefix, message);
    }

    pub fn failed(&self, prefix: &str, message: impl Into<String>) {
        self.err_count.fetch_add(1, Ordering::Relaxed);
        self.write_prefixed(self.display.styles.header(Severity::Error), prefix, message);
    }

    fn write_prefixed(&self, color: &ColorSpec, prefix: &str, message: impl Into<String>) {
        let mut buffer = self.emitter.buffer();
        buffer.set_color(&color).ok();
        write!(&mut buffer, "{:>12} ", prefix).unwrap();
        buffer.reset().ok();
        writeln!(&mut buffer, "{}", message.into()).unwrap();
        self.emitter.print(&buffer).unwrap();
    }

    pub fn diagnostic(&self, severity: Severity) -> InFlightDiagnostic<'_> {
        InFlightDiagnostic::new(self, severity)
    }

    #[inline(always)]
    pub fn emit(&self, diagnostic: &Diagnostic) {
        use libeir_diagnostics::term;

        let mut buffer = self.emitter.buffer();
        term::emit(&mut buffer, &self.display, self.codemap.deref(), diagnostic).unwrap();
        self.emitter.print(&buffer).unwrap();
    }
}

#[inline(always)]
pub fn verbosity_to_severity(v: Verbosity) -> Severity {
    match v {
        Verbosity::Silent => Severity::Bug,
        Verbosity::Error => Severity::Error,
        Verbosity::Warning => Severity::Warning,
        Verbosity::Info => Severity::Note,
        Verbosity::Debug => Severity::Note,
    }
}
