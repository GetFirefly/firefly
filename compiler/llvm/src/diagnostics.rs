use std::ffi::CStr;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr;
use std::slice;

use liblumen_util::diagnostics::DiagnosticsHandler;
use liblumen_util::error::FatalError;

use crate::sys::{self as llvm_sys, LLVMDiagnosticInfo};
use crate::utils::strings::{self, RustString};
use crate::{Context, Value};

extern "C" {
    pub type SMDiagnostic;
}

pub type LLVMDiagnosticHandler = unsafe extern "C" fn(&LLVMDiagnosticInfo, *mut libc::c_void);

pub struct DiagnosticInfo<'a> {
    severity: llvm_sys::LLVMDiagnosticSeverity,
    diagnostic: Diagnostic<'a>,
}
impl<'a> DiagnosticInfo<'a> {
    pub fn report(&self, context: &Context, emitter: &DiagnosticsHandler) {
        use llvm_sys::LLVMDiagnosticSeverity::*;
        match self.severity {
            LLVMDSError => emitter.error(self.diagnostic.format(context)),
            LLVMDSWarning => emitter.warn(self.diagnostic.format(context)),
            LLVMDSRemark => emitter.debug(self.diagnostic.format(context)),
            LLVMDSNote => emitter.note(self.diagnostic.format(context)),
        }
    }

    unsafe fn unpack(di: &'a LLVMDiagnosticInfo) -> Self {
        use llvm_sys::core::LLVMGetDiagInfoSeverity;

        let severity = LLVMGetDiagInfoSeverity(di as *const _ as *mut LLVMDiagnosticInfo);
        let diagnostic = Diagnostic::unpack(di);

        Self {
            severity,
            diagnostic,
        }
    }
}

/// LLVMLumenDiagnosticKind
#[derive(Copy, Clone)]
#[repr(C)]
pub enum DiagnosticKind {
    InlineAsm,
    ResourceLimit,
    StackSize,
    Linker,
    DebugMetadataVersion,
    DebugMetadataInvalid,
    ISelFallback,
    SampleProfile,
    OptimizationRemark,
    OptimizationRemarkMissed,
    OptimizationRemarkAnalysis,
    OptimizationRemarkAnalysisFPCommute,
    OptimizationRemarkAnalysisAliasing,
    OptimizationFailure,
    MachineOptimizationRemark,
    MachineOptimizationRemarkMissed,
    MachineOptimizationRemarkAnalysis,
    MIRParser,
    PGOProfile,
    MisExpect,
    Unsupported,
    Other,
}

#[derive(Copy, Clone)]
pub enum OptimizationDiagnosticKind {
    Remark,
    MachineRemark,
    Missed,
    MachineMissed,
    Analysis,
    MachineAnalysis,
    AnalysisFPCommute,
    AnalysisAliasing,
    Failure,
    RemarkOther,
}
impl OptimizationDiagnosticKind {
    pub fn describe(self) -> &'static str {
        match self {
            Self::Remark | Self::MachineRemark | Self::RemarkOther => "remark",
            Self::Missed | Self::MachineMissed => "missed",
            Self::Analysis | Self::MachineAnalysis => "analysis",
            Self::AnalysisFPCommute => "floating-point",
            Self::AnalysisAliasing => "aliasing",
            Self::Failure => "failure",
        }
    }
}

pub struct OptimizationDiagnostic<'a> {
    pub kind: OptimizationDiagnosticKind,
    pub pass_name: String,
    pub function: Value,
    pub line: libc::c_uint,
    pub column: libc::c_uint,
    pub filename: String,
    pub message: String,
    _marker: PhantomData<&'a LLVMDiagnosticInfo>,
}
impl<'a> OptimizationDiagnostic<'a> {
    unsafe fn unpack(kind: OptimizationDiagnosticKind, di: &'a LLVMDiagnosticInfo) -> Self {
        let mut function = MaybeUninit::<Value>::uninit();
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
                        function.as_mut_ptr(),
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

        let function = function.assume_init();

        let mut filename = filename.unwrap_or_default();
        if filename.is_empty() {
            filename.push_str("<unknown file>");
        }

        Self {
            kind,
            pass_name: pass_name.expect("got a non-UTF8 pass name from LLVM"),
            function,
            line,
            column,
            filename,
            message: message.expect("got a non-UTF8 OptimizationDiagnostic message from LLVM"),
            _marker: PhantomData,
        }
    }

    fn format(&self, _context: &Context) -> String {
        use llvm_sys::core::LLVMGetValueName2;

        let fun = unsafe {
            let mut len = MaybeUninit::<libc::size_t>::uninit();
            let ptr = LLVMGetValueName2(self.function, len.as_mut_ptr());
            let len = len.assume_init();
            let bytes = slice::from_raw_parts(ptr as *const u8, len as usize);
            std::str::from_utf8(bytes).unwrap()
        };
        format!(
            "optimization pass {} ({}) in {} ({}:{} @ {})",
            self.pass_name,
            self.kind.describe(),
            fun,
            self.line,
            self.column,
            self.filename
        )
    }
}

pub struct ISelFallbackDiagnostic<'a> {
    pub function: Value,
    _marker: PhantomData<&'a LLVMDiagnosticInfo>,
}
impl<'a> ISelFallbackDiagnostic<'a> {
    unsafe fn unpack(di: &'a LLVMDiagnosticInfo) -> Self {
        let mut function = MaybeUninit::<Value>::uninit();
        LLVMLumenUnpackISelFallbackDiagnostic(di, function.as_mut_ptr());
        let function = function.assume_init();
        Self {
            function,
            _marker: PhantomData,
        }
    }
}

pub enum Diagnostic<'a> {
    Optimization(OptimizationDiagnostic<'a>),
    Linker(&'a LLVMDiagnosticInfo),
    ISelFallback(ISelFallbackDiagnostic<'a>),
    Unknown(&'a LLVMDiagnosticInfo),
}
impl<'a> Diagnostic<'a> {
    unsafe fn unpack(di: &'a LLVMDiagnosticInfo) -> Self {
        use OptimizationDiagnosticKind::*;

        match LLVMLumenGetDiagInfoKind(di) {
            DiagnosticKind::OptimizationRemark => {
                Self::Optimization(OptimizationDiagnostic::unpack(Remark, di))
            }
            DiagnosticKind::MachineOptimizationRemark => {
                Self::Optimization(OptimizationDiagnostic::unpack(MachineRemark, di))
            }
            DiagnosticKind::OptimizationRemarkMissed => {
                Self::Optimization(OptimizationDiagnostic::unpack(Missed, di))
            }
            DiagnosticKind::MachineOptimizationRemarkMissed => {
                Self::Optimization(OptimizationDiagnostic::unpack(MachineMissed, di))
            }
            DiagnosticKind::OptimizationRemarkAnalysis => {
                Self::Optimization(OptimizationDiagnostic::unpack(Analysis, di))
            }
            DiagnosticKind::MachineOptimizationRemarkAnalysis => {
                Self::Optimization(OptimizationDiagnostic::unpack(MachineAnalysis, di))
            }
            DiagnosticKind::OptimizationRemarkAnalysisFPCommute => {
                Self::Optimization(OptimizationDiagnostic::unpack(AnalysisFPCommute, di))
            }
            DiagnosticKind::OptimizationRemarkAnalysisAliasing => {
                Self::Optimization(OptimizationDiagnostic::unpack(AnalysisAliasing, di))
            }
            DiagnosticKind::OptimizationFailure => {
                Self::Optimization(OptimizationDiagnostic::unpack(Failure, di))
            }
            DiagnosticKind::ISelFallback => Self::ISelFallback(ISelFallbackDiagnostic::unpack(di)),
            DiagnosticKind::Linker => Self::Linker(di),
            _ => Self::Unknown(di),
        }
    }

    fn format(&self, context: &Context) -> String {
        use llvm_sys::core::LLVMGetValueName2;

        match self {
            Self::Optimization(info) => info.format(context),
            Self::ISelFallback(info) => {
                let fun = unsafe {
                    let mut len = MaybeUninit::<libc::size_t>::uninit();
                    let ptr = LLVMGetValueName2(info.function, len.as_mut_ptr());
                    let len = len.assume_init();
                    let bytes = slice::from_raw_parts(ptr as *const u8, len as usize);
                    std::str::from_utf8(bytes).unwrap()
                };
                format!("instruction selection used fallback path in {}", fun)
            }
            Self::Linker(info) => {
                let message = strings::build_string(|message| unsafe {
                    LLVMLumenWriteDiagnosticInfoToString(info, message);
                })
                .ok();
                format!("[linker] {}", message.unwrap_or(String::new()))
            }
            Self::Unknown(info) => strings::build_string(|message| unsafe {
                LLVMLumenWriteDiagnosticInfoToString(info, message);
            })
            .ok()
            .unwrap_or(String::new()),
        }
    }
}

pub fn init() {
    unsafe {
        LLVMLumenInstallFatalErrorHandler();
    }
}

pub(crate) unsafe extern "C" fn diagnostic_handler(
    info: &LLVMDiagnosticInfo,
    user: *mut libc::c_void,
) {
    use std::sync::Weak;

    if user.is_null() {
        return;
    }

    let (context, diagnostics) = &*(user as *const (&Context, Weak<DiagnosticsHandler>));
    if let Some(diagnostics) = diagnostics.upgrade() {
        let info = DiagnosticInfo::unpack(info);

        info.report(context, &diagnostics);
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
    pub fn LLVMLumenGetDiagInfoKind(info: &LLVMDiagnosticInfo) -> DiagnosticKind;
    pub fn LLVMLumenIsVerboseOptimizationDiagnostic(info: &LLVMDiagnosticInfo) -> bool;
    #[allow(improper_ctypes)]
    pub fn LLVMLumenUnpackOptimizationDiagnostic(
        info: &LLVMDiagnosticInfo,
        pass_name_out: &RustString,
        function_out: *mut Value,
        loc_line_out: &mut libc::c_uint,
        loc_column_out: &mut libc::c_uint,
        loc_filename_out: &RustString,
        message_out: &RustString,
    );
    pub fn LLVMLumenUnpackISelFallbackDiagnostic(
        info: &LLVMDiagnosticInfo,
        function_out: *mut Value,
    );
    #[allow(improper_ctypes)]
    pub fn LLVMLumenWriteDiagnosticInfoToString(info: &LLVMDiagnosticInfo, s: &RustString);
    #[allow(improper_ctypes)]
    pub fn LLVMLumenWriteSMDiagnosticToString(d: &SMDiagnostic, s: &RustString);
}
