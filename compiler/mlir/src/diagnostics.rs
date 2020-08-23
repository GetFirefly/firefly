use std::borrow::Cow;
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};

use liblumen_llvm::utils::{strings, RustString};
use liblumen_util::diagnostics::{
    DiagnosticsHandler, FileName, InFlightDiagnostic, LabelStyle, SourceId,
};

use crate::ContextRef;

mod ffi {
    use super::*;
    use liblumen_compiler_macros::foreign_struct;

    #[foreign_struct]
    pub struct DiagnosticEngine;
    #[foreign_struct]
    pub struct DiagnosticInfo;

    impl DiagnosticInfo {
        pub(super) fn severity(&self) -> Severity {
            unsafe { MLIRGetDiagnosticSeverity(self) }
        }

        pub(super) fn location(&self) -> Option<SourceLoc> {
            let mut loc = MaybeUninit::<FileLineColLoc>::uninit();
            if unsafe { MLIRGetDiagnosticLocation(self, loc.as_mut_ptr()) } {
                let loc = unsafe { loc.assume_init() };
                Some(loc.into())
            } else {
                None
            }
        }

        pub(super) fn to_string(&self) -> String {
            strings::build_string(|rs| {
                unsafe { MLIRWriteDiagnosticToString(self, rs) };
            })
        }
    }

    #[repr(u32)]
    pub enum Severity {
        Note = 0,
        Warning,
        Error,
        Remark,
    }

    #[repr(C)]
    pub struct FileLineColLoc {
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
}

pub use self::ffi::{DiagnosticEngine, DiagnosticEngineRef, DiagnosticInfo, DiagnosticInfoRef};

pub type MLIRDiagnosticHandler = extern "C" fn(&DiagnosticInfo, &DiagnosticsHandler) -> bool;
pub type MLIRDiagnosticNoteHandler = extern "C" fn(&DiagnosticInfo, &mut InFlightDiagnostic);

pub struct SourceLoc {
    filename: FileName,
    line: u32,
    column: u32,
}
impl From<ffi::FileLineColLoc> for SourceLoc {
    fn from(loc: ffi::FileLineColLoc) -> Self {
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
pub(crate) extern "C" fn on_diagnostic(
    info: &DiagnosticInfo,
    handler: &DiagnosticsHandler,
) -> bool {
    use liblumen_util::diagnostics::{LabelStyle, Severity};
    let mut ifd = match info.severity() {
        ffi::Severity::Note | ffi::Severity::Remark => handler.diagnostic(Severity::Note),
        ffi::Severity::Warning => handler.diagnostic(Severity::Warning),
        ffi::Severity::Error => handler.diagnostic(Severity::Error),
    };
    let message = info.to_string();
    ifd.with_message(message);
    if let Some(loc) = info.location() {
        ifd.set_source_file(loc.filename);
        ifd.with_primary_label(
            loc.line,
            loc.column,
            Some("problem during codegen related to this source code".to_owned()),
        );
    }
    unsafe {
        MLIRForEachDiagnosticNote(
            info,
            append_note_to_diagnostic,
            &mut ifd as *mut _ as *mut std::ffi::c_void,
        );
    }
    ifd.emit();

    // Do not propagate
    false
}

extern "C" fn append_note_to_diagnostic(note: &DiagnosticInfo, ifd: &mut InFlightDiagnostic) {
    let message = note.to_string();
    let mut buffer = String::with_capacity(message.len() * 2);
    for line in message.lines() {
        buffer.push_str(line);
        buffer.push_str("\n    ");
    }
    buffer.shrink_to_fit();
    if let Some(loc) = note.location() {
        ifd.with_label(
            LabelStyle::Secondary,
            Some(loc.filename),
            loc.line,
            loc.column,
            Some(buffer),
        );
    } else {
        ifd.with_note(buffer);
    }
}

extern "C" {
    #[allow(improper_ctypes)]
    pub fn MLIRGetDiagnosticEngine(context: ContextRef) -> &'static mut DiagnosticEngine;

    #[allow(improper_ctypes)]
    pub fn MLIRRegisterDiagnosticHandler(
        context: ContextRef,
        handler: *const DiagnosticsHandler,
        callback: MLIRDiagnosticHandler,
    );

    #[allow(improper_ctypes)]
    pub fn MLIRGetDiagnosticLocation(
        diagnostic: &DiagnosticInfo,
        loc: *mut ffi::FileLineColLoc,
    ) -> bool;

    #[allow(improper_ctypes)]
    pub fn MLIRGetDiagnosticSeverity(diagnostic: &DiagnosticInfo) -> ffi::Severity;

    #[allow(improper_ctypes)]
    pub fn MLIRForEachDiagnosticNote(
        diagnostic: &DiagnosticInfo,
        callback: MLIRDiagnosticNoteHandler,
        ifd: *mut std::ffi::c_void,
    );

    #[allow(improper_ctypes)]
    pub fn MLIRWriteDiagnosticToString(d: &DiagnosticInfo, s: &RustString);
}
