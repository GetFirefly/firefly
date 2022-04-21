use std::ffi::c_void;
use std::fmt;
use std::path::{Path, PathBuf};

use liblumen_diagnostics::Severity;
use liblumen_util::diagnostics::{DiagnosticsHandler, FileName, InFlightDiagnostic, LabelStyle};

use crate::support::MlirStringCallback;
use crate::*;

/// Register our global diagnostics handler as the diagnostic handler for the given
/// MLIR context. Any MLIR-produced diagnostics will be translated to our own diagnostic
/// system.
pub fn register_diagnostics_handler(context: Context, handler: &DiagnosticsHandler) {
    unsafe {
        mlir_context_attach_diagnostic_handler(
            context,
            on_diagnostic,
            handler as *const DiagnosticsHandler as *const c_void,
            None,
        );
    }
}

extern "C" {
    type MlirDiagnostic;
}

/// This type represents the callback invoked when diagnostics occurs
pub type MlirDiagnosticHandler = extern "C" fn(Diagnostic, *const c_void) -> LogicalResult;

/// This type represents the callback invoked when cleaning up a diagnostic handler
pub type CleanupUserDataFunction = extern "C" fn(*const c_void);

/// This type represents the unique identifier for a diagnostic handler
pub type DiagnosticHandlerId = u64;

/// This type is a wrapper for an MLIR-owned diagnostic, and it exposes
/// functionality necessary to interrogate and print such diagnostics.
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Diagnostic(*mut MlirDiagnostic);
impl Diagnostic {
    #[inline(always)]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    pub fn severity(self) -> Severity {
        match unsafe { mlir_diagnostic_get_severity(self) } {
            MlirSeverity::Error => Severity::Error,
            MlirSeverity::Warning => Severity::Warning,
            MlirSeverity::Note | MlirSeverity::Remark => Severity::Note,
        }
    }

    pub fn location(self) -> Option<SourceLoc> {
        let loc = unsafe { mlir_diagnostic_get_file_line_col(self) };
        if loc.filename.is_null() {
            return None;
        }
        Some(loc.into())
    }

    pub fn notes(self) -> impl Iterator<Item = Diagnostic> {
        let len = unsafe { mlir_diagnostic_get_num_notes(self) };
        DiagnosticNoteIter {
            diag: self,
            len,
            pos: 0,
        }
    }
}
impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe {
            mlir_diagnostic_print(
                *self,
                support::write_to_formatter,
                f as *mut _ as *mut c_void,
            );
        }
        Ok(())
    }
}

/// Iterator for notes attached to a Diagnostic
struct DiagnosticNoteIter {
    diag: Diagnostic,
    len: usize,
    pos: usize,
}
impl Iterator for DiagnosticNoteIter {
    type Item = Diagnostic;

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.len {
            return None;
        }
        let diag = unsafe { mlir_diagnostic_get_note(self.diag, self.pos) };
        if diag.is_null() {
            self.len = 0;
            return None;
        }
        self.pos += 1;
        Some(diag)
    }
}
impl std::iter::FusedIterator for DiagnosticNoteIter {}

/// This is the severity level returned via FFI
///
/// NOTE: rustc complains that the variants are never constructed, which
/// is true for our Rust code, but since it is constructed in C++, we only
/// ever match on it, hence the `#[allow(unused)]` below
#[repr(u32)]
enum MlirSeverity {
    #[allow(unused)]
    Error = 0,
    #[allow(unused)]
    Warning,
    #[allow(unused)]
    Note,
    #[allow(unused)]
    Remark,
}

/// This is the location value returned via FFI
#[repr(C)]
struct FileLineColLoc {
    pub line: u32,
    pub column: u32,
    pub filename_len: u32,
    pub filename: *const u8,
}
impl FileLineColLoc {
    #[inline]
    pub(super) fn filename(&self) -> String {
        let bytes =
            unsafe { core::slice::from_raw_parts(self.filename, self.filename_len as usize) };
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// This is the location value we produce from a `FileLineColLoc`
pub struct SourceLoc {
    filename: FileName,
    line: u32,
    column: u32,
}
impl From<FileLineColLoc> for SourceLoc {
    fn from(loc: FileLineColLoc) -> Self {
        let filename = loc.filename();
        let path = Path::new(&filename);
        if path.exists() {
            SourceLoc {
                filename: FileName::from(PathBuf::from(filename)),
                line: loc.line,
                column: loc.column,
            }
        } else {
            SourceLoc {
                filename: FileName::from(filename),
                line: loc.line,
                column: loc.column,
            }
        }
    }
}

/// This function is used as a callback for MLIR diagnostics
pub(crate) extern "C" fn on_diagnostic(diag: Diagnostic, userdata: *const c_void) -> LogicalResult {
    let handler = unsafe { &*(userdata as *const DiagnosticsHandler) };
    let mut ifd = handler.diagnostic(diag.severity());
    ifd.with_message(diag.to_string());
    if let Some(loc) = diag.location() {
        ifd.set_source_file(loc.filename);
        ifd.with_primary_label(
            loc.line,
            loc.column,
            Some("during codegen of this expression".to_owned()),
        );
    }
    for note in diag.notes() {
        append_note_to_diagnostic(note, &mut ifd);
    }
    ifd.emit();

    LogicalResult::Success
}

fn append_note_to_diagnostic(note: Diagnostic, ifd: &mut InFlightDiagnostic) {
    let message = note.to_string();
    if let Some(loc) = note.location() {
        ifd.with_label(
            LabelStyle::Secondary,
            Some(loc.filename),
            loc.line,
            loc.column,
            Some(message),
        );
    } else {
        ifd.with_note(message);
    }
}

extern "C" {
    #[link_name = "mlirContextAttachDiagnosticHandler"]
    fn mlir_context_attach_diagnostic_handler(
        context: Context,
        callback: MlirDiagnosticHandler,
        userdata: *const c_void,
        cleanup: Option<CleanupUserDataFunction>,
    ) -> DiagnosticHandlerId;

    #[allow(unused)]
    #[link_name = "mlirContextDetachDiagnosticHandler"]
    fn mlir_context_detach_diagnostic_handler(context: Context, id: DiagnosticHandlerId);

    #[link_name = "mlirDiagnosticGetSeverity"]
    fn mlir_diagnostic_get_severity(diag: Diagnostic) -> MlirSeverity;

    #[link_name = "mlirDiagnosticGetNumNotes"]
    fn mlir_diagnostic_get_num_notes(diag: Diagnostic) -> usize;

    #[link_name = "mlirDiagnosticGetNote"]
    fn mlir_diagnostic_get_note(diag: Diagnostic, index: usize) -> Diagnostic;

    #[link_name = "mlirDiagnosticPrint"]
    fn mlir_diagnostic_print(diag: Diagnostic, callback: MlirStringCallback, userdata: *mut c_void);

    #[link_name = "mlirDiagnosticGetFileLineCol"]
    fn mlir_diagnostic_get_file_line_col(diag: Diagnostic) -> FileLineColLoc;
}
