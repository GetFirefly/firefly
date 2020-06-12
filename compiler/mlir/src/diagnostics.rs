use liblumen_llvm::utils::RustString;
use liblumen_util::diagnostics::DiagnosticsHandler;

use crate::Context;

mod ffi {
    use liblumen_compiler_macros::foreign_struct;

    #[foreign_struct]
    pub struct DiagnosticEngine;
    #[foreign_struct]
    pub struct DiagnosticInfo;
}

pub use self::ffi::{DiagnosticEngine, DiagnosticEngineRef, DiagnosticInfo, DiagnosticInfoRef};

pub type MLIRDiagnosticHandler = extern "C" fn(&DiagnosticInfo, *mut libc::c_void);

extern "C" {
    #[allow(improper_ctypes)]
    pub fn MLIRGetDiagnosticEngine(context: &Context) -> &'static mut DiagnosticEngine;

    #[allow(improper_ctypes)]
    pub fn MLIRRegisterDiagnosticHandler(
        context: &Context,
        handler: *const DiagnosticsHandler,
        callback: &MLIRDiagnosticHandler,
    );

    #[allow(improper_ctypes)]
    pub fn MLIRWriteDiagnosticToString(d: &DiagnosticInfo, s: &RustString);
}
