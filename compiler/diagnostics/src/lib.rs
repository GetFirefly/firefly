mod codemap;
mod filename;
mod index;
mod source;
mod span;

pub use codespan::Location;
pub use codespan::{ByteIndex, ByteOffset};
pub use codespan::{ColumnIndex, ColumnNumber, ColumnOffset};
pub use codespan::{Index, Offset};
pub use codespan::{LineIndex, LineNumber, LineOffset};
pub use codespan::{RawIndex, RawOffset};

pub use codespan_reporting::diagnostic::{LabelStyle, Severity};
pub use codespan_reporting::files::{Error, Files};
pub use codespan_reporting::term;

pub use liblumen_diagnostics_macros::*;

pub use self::codemap::CodeMap;
pub use self::filename::FileName;
pub use self::index::SourceIndex;
pub use self::source::{SourceFile, SourceId};
pub use self::span::{SourceSpan, Span, Spanned};

pub type Diagnostic = codespan_reporting::diagnostic::Diagnostic<SourceId>;
pub type Label = codespan_reporting::diagnostic::Label<SourceId>;

pub trait ToDiagnostic {
    fn to_diagnostic(&self) -> Diagnostic;
}

use std::cell::{Ref, RefCell};
use std::rc::Rc;

#[derive(Default, Clone)]
pub struct Reporter(Rc<RefCell<ReporterImpl>>);
impl Reporter {
    /// Creates a new reporter with default configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new reporter which will ignore any diagnostic reports
    pub fn null() -> Self {
        Self(Rc::new(RefCell::new(ReporterImpl::new(false, true))))
    }

    /// Creates a new reporter with warnings-as-errors enabled
    pub fn strict() -> Self {
        Self(Rc::new(RefCell::new(ReporterImpl::new(true, false))))
    }

    /// Set whether or not warnings will be treated as errors by the reporter
    ///
    /// When true, any warning diagnostics will be automatically promoted to errors
    pub fn warnings_as_errors(&self, value: bool) {
        let mut reporter = self.0.borrow_mut();
        reporter.warnings_as_errors(value);
    }

    /// Set whether or not this reporter will gather any diagnostics
    ///
    /// When silent, any diagnostics reported are silently dropped
    pub fn silence(&self, value: bool) {
        let mut reporter = self.0.borrow_mut();
        reporter.silence(value);
    }

    /// Returns true if an error was reported
    #[inline]
    pub fn is_failed(&self) -> bool {
        let reporter = self.0.borrow();
        reporter.is_failed()
    }

    /// Print all diagnostics, using the provided CodeMap to load sources
    pub fn print(&self, codemap: &CodeMap) {
        let reporter = self.0.borrow();
        reporter.print(codemap);
    }

    /// Get a slice of all the diagnostics reported since creation
    pub fn diagnostics(&self) -> Ref<'_, [Diagnostic]> {
        Ref::map(self.0.borrow(), |r| r.diagnostics())
    }

    /// Report a diagnostic
    #[inline]
    pub fn diagnostic(&self, diagnostic: Diagnostic) {
        let mut reporter = self.0.borrow_mut();
        reporter.diagnostic(diagnostic);
    }

    /// Report a diagnostic, forcing its severity to Warning
    pub fn warning<W: ToDiagnostic>(&self, warning: W) {
        let mut diagnostic = warning.to_diagnostic();
        diagnostic.severity = Severity::Warning;
        let mut reporter = self.0.borrow_mut();
        reporter.diagnostic(diagnostic)
    }

    /// Report a diagnostic, forcing its severity to Error
    pub fn error<E: ToDiagnostic>(&self, error: E) {
        let mut diagnostic = error.to_diagnostic();
        diagnostic.severity = Severity::Error;
        let mut reporter = self.0.borrow_mut();
        reporter.diagnostic(diagnostic)
    }

    /// A convenience method to make expressing common error diagnostics easier
    pub fn show_error(&self, message: &str, labels: &[(SourceSpan, &str)]) {
        if labels.is_empty() {
            self.diagnostic(Diagnostic::error().with_message(message));
        } else {
            let labels = labels
                .iter()
                .copied()
                .enumerate()
                .map(|(i, (span, message))| {
                    if i > 0 {
                        Label::secondary(span.source_id(), span).with_message(message)
                    } else {
                        Label::primary(span.source_id(), span).with_message(message)
                    }
                })
                .collect();
            self.diagnostic(
                Diagnostic::error()
                    .with_message(message)
                    .with_labels(labels),
            );
        }
    }

    /// A convenience method to make expressing common warning diagnostics easier
    pub fn show_warning(&mut self, message: &str, labels: &[(SourceSpan, &str)]) {
        if labels.is_empty() {
            self.diagnostic(Diagnostic::warning().with_message(message));
        } else {
            let labels = labels
                .iter()
                .copied()
                .enumerate()
                .map(|(i, (span, message))| {
                    if i > 0 {
                        Label::secondary(span.source_id(), span).with_message(message)
                    } else {
                        Label::primary(span.source_id(), span).with_message(message)
                    }
                })
                .collect();
            self.diagnostic(
                Diagnostic::warning()
                    .with_message(message)
                    .with_labels(labels),
            );
        }
    }
}

#[derive(Default, Clone)]
pub struct ReporterImpl {
    diagnostics: Vec<Diagnostic>,
    warnings_as_errors: bool,
    failed: bool,
    silent: bool,
}
impl ReporterImpl {
    fn new(warnings_as_errors: bool, silent: bool) -> Self {
        Self {
            diagnostics: vec![],
            warnings_as_errors,
            failed: false,
            silent,
        }
    }

    fn warnings_as_errors(&mut self, value: bool) {
        self.warnings_as_errors = value;
    }

    fn silence(&mut self, value: bool) {
        self.silent = value;
    }

    fn is_failed(&self) -> bool {
        self.failed
    }

    fn print(&self, codemap: &CodeMap) {
        use term::termcolor::{ColorChoice, StandardStream};
        use term::Config;
        let config = Config::default();
        let mut out = StandardStream::stderr(ColorChoice::Auto);
        for diag in &self.diagnostics {
            term::emit(&mut out, &config, codemap, &diag).unwrap();
        }
    }

    fn diagnostics(&self) -> &[Diagnostic] {
        self.diagnostics.as_slice()
    }

    fn diagnostic(&mut self, diagnostic: Diagnostic) {
        if !self.silent {
            match diagnostic.severity {
                Severity::Bug | Severity::Error => {
                    self.diagnostics.push(diagnostic);
                    self.failed = true;
                }
                Severity::Warning if self.warnings_as_errors => {
                    self.diagnostics.push(diagnostic);
                    self.failed = true;
                }
                _ => self.diagnostics.push(diagnostic),
            }
        }
    }
}
