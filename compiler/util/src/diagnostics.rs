use std::io::Write;
use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

pub type DisplayConfig = firefly_diagnostics::term::Config;
pub type DisplayStyle = firefly_diagnostics::term::DisplayStyle;
pub type DisplayChars = firefly_diagnostics::term::Chars;

pub use firefly_diagnostics::term::termcolor::*;
pub use firefly_diagnostics::term::{ColorArg, Styles};
pub use firefly_diagnostics::{
    ByteIndex, CodeMap, FileName, Files, SourceFile, SourceId, SourceIndex, SourceSpan, Span,
    Spanned, ToDiagnostic,
};
pub use firefly_diagnostics::{Diagnostic, Label, LabelStyle, Severity};

use crate::error::{FatalError, Verbosity};

#[derive(Debug, Clone)]
pub struct DiagnosticsConfig {
    pub verbosity: Verbosity,
    pub warnings_as_errors: bool,
    pub no_warn: bool,
    pub display: DisplayConfig,
}
impl Default for DiagnosticsConfig {
    fn default() -> Self {
        Self {
            verbosity: Verbosity::Info,
            warnings_as_errors: false,
            no_warn: false,
            display: DisplayConfig::default(),
        }
    }
}

pub trait Emitter: Send + Sync {
    fn buffer(&self) -> Buffer;
    fn print(&self, buffer: Buffer) -> std::io::Result<()>;
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
    fn print(&self, buffer: Buffer) -> std::io::Result<()> {
        self.writer.print(&buffer)
    }
}

#[derive(Default)]
pub struct CaptureEmitter {
    buffer: Mutex<Vec<u8>>,
}
impl CaptureEmitter {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn captured(&self) -> String {
        let buf = self.buffer.lock();
        String::from_utf8_lossy(buf.as_slice()).into_owned()
    }
}
impl Emitter for CaptureEmitter {
    #[inline]
    fn buffer(&self) -> Buffer {
        Buffer::no_color()
    }

    #[inline]
    fn print(&self, buffer: Buffer) -> std::io::Result<()> {
        let mut bytes = buffer.into_inner();
        let mut buf = self.buffer.lock();
        buf.append(&mut bytes);
        Ok(())
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
    fn print(&self, _buffer: Buffer) -> std::io::Result<()> {
        Ok(())
    }
}

/// Construct an in-flight diagnostic
pub struct InFlightDiagnostic<'h> {
    handler: &'h DiagnosticsHandler,
    file_id: Option<SourceId>,
    diagnostic: Diagnostic,
    severity: Severity,
}
impl<'h> InFlightDiagnostic<'h> {
    fn new(handler: &'h DiagnosticsHandler, severity: Severity) -> Self {
        Self {
            handler,
            file_id: None,
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
        use firefly_diagnostics::term::DisplayStyle;
        match self.handler.display.display_style {
            DisplayStyle::Rich => true,
            _ => false,
        }
    }

    pub fn set_source_file(mut self, filename: impl Into<FileName>) -> Self {
        let filename = filename.into();
        let file_id = self.handler.codemap.get_file_id(&filename);
        self.file_id = file_id;
        self
    }

    pub fn with_message(mut self, message: impl ToString) -> Self {
        self.diagnostic.message = message.to_string();
        self
    }

    pub fn with_primary_span(mut self, span: SourceSpan) -> Self {
        self.diagnostic
            .labels
            .push(Label::primary(span.source_id(), span));
        self
    }

    pub fn with_primary_label(mut self, span: SourceSpan, message: impl ToString) -> Self {
        self.diagnostic
            .labels
            .push(Label::primary(span.source_id(), span).with_message(message.to_string()));
        self
    }

    pub fn with_secondary_label(mut self, span: SourceSpan, message: impl ToString) -> Self {
        self.diagnostic
            .labels
            .push(Label::secondary(span.source_id(), span).with_message(message.to_string()));
        self
    }

    pub fn with_primary_label_line_and_col(
        self,
        line: u32,
        column: u32,
        message: Option<String>,
    ) -> Self {
        let file_id = self.file_id.clone();
        self.with_label_and_file_id(LabelStyle::Primary, file_id, line, column, message)
    }

    pub fn with_label(
        self,
        style: LabelStyle,
        filename: Option<FileName>,
        line: u32,
        column: u32,
        message: Option<String>,
    ) -> Self {
        if let Some(name) = filename {
            let id = self.handler.lookup_file_id(name);
            self.with_label_and_file_id(style, id, line, column, message)
        } else {
            self
        }
    }

    fn with_label_and_file_id(
        mut self,
        style: LabelStyle,
        file_id: Option<SourceId>,
        line: u32,
        _column: u32,
        message: Option<String>,
    ) -> Self {
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
            self
        } else {
            self
        }
    }

    pub fn with_note(mut self, note: impl ToString) -> Self {
        self.diagnostic.notes.push(note.to_string());
        self
    }

    pub fn add_note(&mut self, note: impl ToString) {
        self.diagnostic.notes.push(note.to_string());
    }

    pub fn take(self) -> Diagnostic {
        self.diagnostic
    }

    /// Emit the diagnostic via the DiagnosticHandler
    pub fn emit(self) {
        self.handler.emit(self.diagnostic);
    }
}

pub struct DiagnosticsHandler {
    emitter: Arc<dyn Emitter>,
    codemap: Arc<CodeMap>,
    err_count: AtomicUsize,
    verbosity: Verbosity,
    warnings_as_errors: bool,
    no_warn: bool,
    silent: bool,
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
        let no_warn = config.no_warn || config.verbosity > Verbosity::Warning;
        Self {
            emitter,
            codemap,
            err_count: AtomicUsize::new(0),
            verbosity: config.verbosity,
            warnings_as_errors: config.warnings_as_errors,
            no_warn,
            silent: config.verbosity == Verbosity::Silent,
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

    /// Emits an error message and produces a FatalError object
    /// which can be used to terminate execution immediately
    pub fn fatal(&self, err: impl ToString) -> FatalError {
        self.error(err);
        FatalError
    }

    /// Report a diagnostic, forcing its severity to Error
    pub fn error(&self, error: impl ToString) {
        self.err_count.fetch_add(1, Ordering::Relaxed);
        let diagnostic = Diagnostic::error().with_message(error.to_string());
        self.emit(diagnostic);
    }

    /// Report a diagnostic, forcing its severity to Warning
    pub fn warn(&self, warning: impl ToString) {
        if self.warnings_as_errors {
            return self.error(warning);
        }
        let diagnostic = Diagnostic::warning().with_message(warning.to_string());
        self.emit(diagnostic);
    }

    /// Emits an informational message
    pub fn info(&self, message: impl ToString) {
        if self.verbosity > Verbosity::Info {
            return;
        }
        let info_color = self.display.styles.header(Severity::Help);
        let mut buffer = self.emitter.buffer();
        buffer.set_color(&info_color).ok();
        write!(&mut buffer, "info").unwrap();
        buffer.set_color(&self.display.styles.header_message).ok();
        write!(&mut buffer, ": {}", message.to_string()).unwrap();
        buffer.reset().ok();
        write!(&mut buffer, "\n").unwrap();
        self.emitter.print(buffer).unwrap();
    }

    /// Emits a debug message
    pub fn debug(&self, message: impl ToString) {
        if self.verbosity > Verbosity::Debug {
            return;
        }
        let mut debug_color = self.display.styles.header_message.clone();
        debug_color.set_fg(Some(Color::Blue));
        let mut buffer = self.emitter.buffer();
        buffer.set_color(&debug_color).ok();
        write!(&mut buffer, "debug").unwrap();
        buffer.set_color(&self.display.styles.header_message).ok();
        write!(&mut buffer, ": {}", message.to_string()).unwrap();
        buffer.reset().ok();
        write!(&mut buffer, "\n").unwrap();
        self.emitter.print(buffer).unwrap();
    }

    /// Emits a note
    pub fn note(&self, message: impl ToString) {
        if self.verbosity > Verbosity::Info {
            return;
        }
        self.emit(Diagnostic::note().with_message(message.to_string()));
    }

    /// Prints a warning-like message with the given prefix
    ///
    /// NOTE: This does not get promoted to an error if warnings-as-errors is set,
    /// as it is intended for informational purposes, not issues with the code being compiled
    pub fn notice(&self, prefix: &str, message: impl ToString) {
        if self.verbosity > Verbosity::Info {
            return;
        }
        self.write_prefixed(
            self.display.styles.header(Severity::Warning),
            prefix,
            message,
        );
    }

    /// Prints a success message with the given prefix
    pub fn success(&self, prefix: &str, message: impl ToString) {
        if self.silent {
            return;
        }
        self.write_prefixed(self.display.styles.header(Severity::Note), prefix, message);
    }

    /// Prints an error message with the given prefix
    pub fn failed(&self, prefix: &str, message: impl ToString) {
        self.err_count.fetch_add(1, Ordering::Relaxed);
        self.write_prefixed(self.display.styles.header(Severity::Error), prefix, message);
    }

    fn write_prefixed(&self, color: &ColorSpec, prefix: &str, message: impl ToString) {
        let mut buffer = self.emitter.buffer();
        buffer.set_color(&color).ok();
        write!(&mut buffer, "{:>12} ", prefix).unwrap();
        buffer.reset().ok();
        writeln!(&mut buffer, "{}", message.to_string()).unwrap();
        self.emitter.print(buffer).unwrap();
    }

    /// Generates an in-flight diagnostic for more complex diagnostics use cases
    ///
    /// The caller is responsible for dropping/emitting the diagnostic using the in-flight APIs
    pub fn diagnostic(&self, severity: Severity) -> InFlightDiagnostic<'_> {
        InFlightDiagnostic::new(self, severity)
    }

    /// Emits the given diagnostic
    #[inline(always)]
    pub fn emit(&self, diagnostic: impl ToDiagnostic) {
        use firefly_diagnostics::term;

        if self.silent {
            return;
        }

        let mut diagnostic = diagnostic.to_diagnostic();
        match diagnostic.severity {
            Severity::Note if self.verbosity > Verbosity::Info => return,
            Severity::Warning if self.no_warn => return,
            Severity::Warning if self.warnings_as_errors => {
                diagnostic.severity = Severity::Error;
            }
            _ => (),
        }

        let mut buffer = self.emitter.buffer();
        term::emit(
            &mut buffer,
            &self.display,
            self.codemap.deref(),
            &diagnostic,
        )
        .unwrap();
        self.emitter.print(buffer).unwrap();
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
