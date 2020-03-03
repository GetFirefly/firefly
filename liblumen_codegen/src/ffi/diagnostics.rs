use liblumen_llvm::string::RustString;
use liblumen_session::DiagnosticsHandler;

use crate::mlir::builder::ffi::foreign_types as mlir;

pub type MLIRDiagnosticHandler = extern "C" fn(&mlir::DiagnosticInfo, *mut libc::c_void);

extern "C" {
    #[allow(improper_ctypes)]
    pub fn MLIRGetDiagnosticEngine(context: &mlir::Context) -> &'static mut mlir::DiagnosticEngine;
    #[allow(improper_ctypes)]
    pub fn MLIRRegisterDiagnosticHandler(
        context: &mlir::Context,
        handler: *const DiagnosticsHandler,
        callback: &MLIRDiagnosticHandler,
    );

    #[allow(improper_ctypes)]
    pub fn MLIRWriteDiagnosticToString(d: &mlir::DiagnosticInfo, s: &RustString);
}
