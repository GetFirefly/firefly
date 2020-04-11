use std::error::Error;
use std::ffi::CString;
use std::fmt::Display;
use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use libeir_diagnostics::emitter::{cyan, green_bold, red_bold, white, yellow, yellow_bold};
use libeir_diagnostics::{ByteSpan, CodeMap, ColorSpec, Diagnostic, Emitter, Severity};
use liblumen_util::error::{FatalError, Verbosity};

#[derive(Debug, Copy, Clone)]
pub struct DiagnosticsConfig {
    pub warnings_as_errors: bool,
    pub no_warn: bool,
}

#[repr(C)]
pub struct Location {
    pub file: CString,
    pub line: u32,
    pub column: u32,
}
impl Location {
    pub fn new(file: CString, line: u32, column: u32) -> Self {
        Self { file, line, column }
    }
}

#[derive(Clone)]
pub struct DiagnosticsHandler {
    emitter: Arc<dyn Emitter>,
    codemap: Arc<RwLock<CodeMap>>,
    warnings_as_errors: bool,
    no_warn: bool,
    err_count: Arc<AtomicUsize>,
}
// We can safely implement these traits for DiagnosticsHandler,
// as the only two non-atomic fields are read-only after creation
unsafe impl Send for DiagnosticsHandler {}
unsafe impl Sync for DiagnosticsHandler {}
impl DiagnosticsHandler {
    pub fn new(
        config: DiagnosticsConfig,
        codemap: Arc<RwLock<CodeMap>>,
        emitter: Arc<dyn Emitter>,
    ) -> Self {
        Self {
            emitter,
            codemap,
            warnings_as_errors: config.warnings_as_errors,
            no_warn: config.no_warn,
            err_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn location(&self, span: ByteSpan) -> Option<Location> {
        let codemap = self.codemap.read().unwrap();
        let start = span.start();
        codemap
            .find_file(start)
            .map(|fm| {
                (
                    CString::new(fm.name.to_string()).unwrap(),
                    fm.location(start).unwrap(),
                )
            })
            .map(|(file, (li, ci))| {
                Location::new(
                    file,
                    li.number().to_usize() as u32,
                    ci.number().to_usize() as u32,
                )
            })
    }

    pub fn has_errors(&self) -> bool {
        self.err_count.load(Ordering::Relaxed) > 0
    }

    pub fn abort_if_errors(&self) {
        if self.has_errors() {
            FatalError.raise();
        }
    }

    pub fn fatal<E>(&self, err: E) -> FatalError
    where
        E: Deref<Target = (dyn Error + Send + Sync + 'static)>,
    {
        self.write_error(err);
        FatalError
    }

    pub fn fatal_str(&self, err: &str) -> FatalError {
        self.emitter
            .diagnostic(&Diagnostic::new(Severity::Error, err.to_string()))
            .unwrap();
        FatalError
    }

    pub fn error<E>(&self, err: E)
    where
        E: Deref<Target = (dyn Error + Send + Sync + 'static)>,
    {
        self.err_count.fetch_add(1, Ordering::Relaxed);
        self.write_error(err);
    }

    pub fn io_error(&self, err: std::io::Error) {
        self.err_count.fetch_add(1, Ordering::Relaxed);
        let e: &(dyn std::error::Error + Send + Sync + 'static) = &err;
        self.write_error(e);
    }

    pub fn error_str(&self, err: &str) {
        self.err_count.fetch_add(1, Ordering::Relaxed);
        self.emitter
            .diagnostic(&Diagnostic::new(Severity::Error, err.to_string()))
            .unwrap();
    }

    pub fn warn<M: Display>(&self, message: M) {
        if self.warnings_as_errors {
            self.emitter
                .diagnostic(&Diagnostic::new(Severity::Error, message.to_string()))
                .unwrap();
        } else if !self.no_warn {
            self.write_warning(yellow_bold(), "WARN: ");
            self.write_warning(yellow(), message);
        }
    }

    pub fn success<M: Display>(&self, prefix: &str, message: M) {
        self.write_prefixed(green_bold(), prefix, message);
    }

    pub fn failed<M: Display>(&self, prefix: &str, message: M) {
        self.err_count.fetch_add(1, Ordering::Relaxed);
        self.write_prefixed(red_bold(), prefix, message);
    }

    pub fn info<M: Display>(&self, message: M) {
        self.write_info(cyan(), message);
    }

    pub fn debug<M: Display>(&self, message: M) {
        self.write_debug(white(), message);
    }

    pub fn diagnostic(&self, diagnostic: &Diagnostic) {
        self.emitter.diagnostic(diagnostic).unwrap();
    }

    fn write_error<E>(&self, err: E)
    where
        E: Deref<Target = (dyn Error + Send + Sync + 'static)>,
    {
        self.emitter.error(err.deref()).unwrap();
    }

    fn write_warning<M: Display>(&self, color: ColorSpec, message: M) {
        self.emitter
            .warn(Some(color), &message.to_string())
            .unwrap();
    }

    fn write_prefixed<M: Display>(&self, color: ColorSpec, prefix: &str, message: M) {
        self.emitter
            .emit(Some(color), &format!("{:>12} ", prefix))
            .unwrap();
        self.emitter.emit(None, &format!("{}\n", message)).unwrap()
    }

    fn write_info<M: Display>(&self, color: ColorSpec, message: M) {
        self.emitter
            .emit(Some(color), &message.to_string())
            .unwrap();
    }

    fn write_debug<M: Display>(&self, color: ColorSpec, message: M) {
        self.emitter
            .debug(Some(color), &message.to_string())
            .unwrap();
    }
}

pub fn verbosity_to_severity(v: Verbosity) -> Severity {
    match v {
        Verbosity::Silent => Severity::Bug,
        Verbosity::Error => Severity::Error,
        Verbosity::Warning => Severity::Warning,
        Verbosity::Info => Severity::Note,
        Verbosity::Debug => Severity::Note,
    }
}
