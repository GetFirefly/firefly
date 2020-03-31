use std::ffi::CStr;

use liblumen_session::DiagnosticsHandler;
use liblumen_util::error::FatalError;

use crate::utils::strings::{self, RustString};
use crate::Value;

extern "C" {
    pub type SMDiagnostic;
}

pub type DiagnosticInfo = llvm_sys::LLVMDiagnosticInfo;
pub type LLVMDiagnosticHandler = unsafe extern "C" fn(&DiagnosticInfo, *mut libc::c_void);

/// LLVMLumenDiagnosticKind
#[derive(Copy, Clone)]
#[repr(C)]
#[allow(dead_code)] // Variants constructed by C++.
pub enum DiagnosticKind {
    Other,
    InlineAsm,
    StackSize,
    DebugMetadataVersion,
    SampleProfile,
    OptimizationRemark,
    OptimizationRemarkMissed,
    OptimizationRemarkAnalysis,
    OptimizationRemarkAnalysisFPCommute,
    OptimizationRemarkAnalysisAliasing,
    OptimizationRemarkOther,
    OptimizationFailure,
    PGOProfile,
    Linker,
}

#[derive(Copy, Clone)]
pub enum OptimizationDiagnosticKind {
    Remark,
    Missed,
    Analysis,
    AnalysisFPCommute,
    AnalysisAliasing,
    Failure,
    RemarkOther,
}
impl OptimizationDiagnosticKind {
    pub fn describe(self) -> &'static str {
        match self {
            Self::Remark | Self::RemarkOther => "remark",
            Self::Missed => "missed",
            Self::Analysis => "analysis",
            Self::AnalysisFPCommute => "floating-point",
            Self::AnalysisAliasing => "aliasing",
            Self::Failure => "failure",
        }
    }
}

pub struct OptimizationDiagnostic<'a> {
    pub kind: OptimizationDiagnosticKind,
    pub pass_name: String,
    pub function: &'a Value,
    pub line: libc::c_uint,
    pub column: libc::c_uint,
    pub filename: String,
    pub message: String,
}
impl<'a> OptimizationDiagnostic<'a> {
    #[allow(unused)]
    unsafe fn unpack(kind: OptimizationDiagnosticKind, di: &'a DiagnosticInfo) -> Self {
        let mut function: Option<&Value> = None;
        let mut line = 0;
        let mut column = 0;

        let mut message = None;
        let mut filename = None;
        let pass_name = strings::build_string(|pass_name| {
            message = strings::build_string(|message| {
                filename = strings::build_string(|filename| {
                    LLVMLumenUnpackOptimizationDiagnostic(
                        di,
                        pass_name,
                        &mut function,
                        &mut line,
                        &mut column,
                        filename,
                        message,
                    )
                })
                .ok()
            })
            .ok()
        })
        .ok();

        let mut filename = filename.unwrap_or_default();
        if filename.is_empty() {
            filename.push_str("<unknown file>");
        }

        Self {
            kind,
            pass_name: pass_name.expect("got a non-UTF8 pass name from LLVM"),
            function: function.unwrap(),
            line,
            column,
            filename,
            message: message.expect("got a non-UTF8 OptimizationDiagnostic message from LLVM"),
        }
    }
}

pub enum Diagnostic<'a> {
    Optimization(OptimizationDiagnostic<'a>),
    Linker(&'a DiagnosticInfo),
    Unknown(&'a DiagnosticInfo),
}

pub fn init() {
    unsafe {
        LLVMLumenInstallFatalErrorHandler();
    }
}

pub fn fatal_error(handler: &DiagnosticsHandler, msg: &str) -> FatalError {
    match last_error() {
        Some(err) => handler.fatal_str(&format!("{}: {}", msg, err)),
        None => handler.fatal_str(&msg),
    }
}

pub fn last_error() -> Option<String> {
    unsafe {
        let cstr = LLVMLumenGetLastError();
        if cstr.is_null() {
            None
        } else {
            let err = CStr::from_ptr(cstr).to_bytes();
            let err = String::from_utf8_lossy(err).to_string();
            libc::free(cstr as *mut _);
            Some(err)
        }
    }
}

extern "C" {
    pub fn LLVMLumenInstallFatalErrorHandler();
    /// Returns a string describing the last error caused by an LLVM call
    pub fn LLVMLumenGetLastError() -> *const libc::c_char;
    #[allow(improper_ctypes)]
    pub fn LLVMLumenUnpackOptimizationDiagnostic(
        DI: &DiagnosticInfo,
        pass_name_out: &RustString,
        function_out: &mut Option<&Value>,
        loc_line_out: &mut libc::c_uint,
        loc_column_out: &mut libc::c_uint,
        loc_filename_out: &RustString,
        message_out: &RustString,
    );

    #[allow(improper_ctypes)]
    pub fn LLVMLumenWriteDiagnosticInfoToString(DI: &DiagnosticInfo, s: &RustString);
    pub fn LLVMLumenGetDiagInfoKind(DI: &DiagnosticInfo) -> DiagnosticKind;
    #[allow(improper_ctypes)]
    pub fn LLVMLumenWriteSMDiagnosticToString(d: &SMDiagnostic, s: &RustString);
}
