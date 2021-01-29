use std::ffi::CStr;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::slice;

use liblumen_session::Options;
use liblumen_util::diagnostics::{DiagnosticsHandler, FileName, InFlightDiagnostic, Severity};

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
    pub fn report(&self, context: &Context, options: &Options, emitter: &DiagnosticsHandler) {
        use llvm_sys::LLVMDiagnosticSeverity::*;

        // If optimization remarks weren't requested, don't bother with them at all
        if self.severity == LLVMDSRemark && !options.debugging_opts.print_llvm_optimization_remarks
        {
            return;
        }

        let mut ifd = match self.severity {
            LLVMDSNote | LLVMDSRemark => emitter.diagnostic(Severity::Note),
            LLVMDSWarning => emitter.diagnostic(Severity::Warning),
            LLVMDSError => emitter.diagnostic(Severity::Error),
        };

        let should_emit = self.diagnostic.format(&mut ifd, context, options);
        if should_emit {
            ifd.emit();
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

pub struct SourceLoc {
    filename: FileName,
    line: u32,
    column: u32,
}
impl SourceLoc {
    fn new(line: u32, column: u32, filename: String) -> Self {
        let path = Path::new(&filename);
        if path.exists() {
            Self {
                filename: FileName::from(PathBuf::from(filename)),
                line,
                column,
            }
        } else {
            Self {
                filename: FileName::from(filename),
                line,
                column,
            }
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

#[derive(Copy, Clone, PartialEq)]
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
impl From<DiagnosticKind> for OptimizationDiagnosticKind {
    fn from(kind: DiagnosticKind) -> Self {
        use DiagnosticKind::*;
        match kind {
            OptimizationRemark => Self::Remark,
            OptimizationRemarkMissed => Self::Missed,
            OptimizationRemarkAnalysis => Self::Analysis,
            OptimizationRemarkAnalysisFPCommute => Self::AnalysisFPCommute,
            OptimizationRemarkAnalysisAliasing => Self::AnalysisAliasing,
            OptimizationFailure => Self::Failure,
            MachineOptimizationRemark => Self::MachineRemark,
            MachineOptimizationRemarkMissed => Self::MachineMissed,
            MachineOptimizationRemarkAnalysis => Self::MachineAnalysis,
            _ => unreachable!(),
        }
    }
}

pub struct OptimizationDiagnostic<'a> {
    pub kind: OptimizationDiagnosticKind,
    pub pass_name: String,
    pub remark_name: Option<String>,
    pub function: Value,
    pub code_region: Option<Value>,
    pub loc: Option<SourceLoc>,
    pub message: String,
    pub is_verbose: bool,
    _marker: PhantomData<&'a LLVMDiagnosticInfo>,
}
impl<'a> OptimizationDiagnostic<'a> {
    unsafe fn unpack(kind: OptimizationDiagnosticKind, di: &'a LLVMDiagnosticInfo) -> Self {
        let mut function = MaybeUninit::<Value>::uninit();
        let mut code_region = MaybeUninit::<Value>::uninit();
        let mut line = 0;
        let mut column = 0;
        let mut is_verbose = false;

        let mut message = String::new();
        let mut remark_name = None;
        let mut filename = None;
        let pass_name = strings::build_string(|pass_name| {
            remark_name = strings::build_string(|remark_name| {
                message = strings::build_string(|message| {
                    filename = strings::build_string(|filename| {
                        LLVMLumenUnpackOptimizationDiagnostic(
                            di,
                            pass_name,
                            remark_name,
                            function.as_mut_ptr(),
                            code_region.as_mut_ptr(),
                            &mut line,
                            &mut column,
                            &mut is_verbose,
                            filename,
                            message,
                        )
                    })
                })
                .expect("expected diagnostic message, but got an empty string")
            })
        })
        .expect("expected pass name, but got an empty string");

        let function = function.assume_init();
        let code_region = {
            let cr = code_region.assume_init();
            if cr.is_null() {
                None
            } else {
                Some(cr)
            }
        };

        let loc = filename.map(|f| SourceLoc::new(line as u32, column as u32, f));

        Self {
            kind,
            pass_name,
            remark_name,
            function,
            code_region,
            loc,
            message,
            is_verbose,
            _marker: PhantomData,
        }
    }

    fn format(&self, ifd: &mut InFlightDiagnostic, _context: &Context, options: &Options) -> bool {
        use llvm_sys::core::LLVMPrintValueToString;

        // Don't print noisy remarks
        if self.is_verbose {
            return false;
        }

        // Don't print optimization diagnostics other than failures, unless explicitly requested
        if self.kind != OptimizationDiagnosticKind::Failure
            && !options.debugging_opts.print_llvm_optimization_remarks
        {
            return false;
        }

        ifd.with_message(format!("optimization {}", self.kind.describe()));

        if let Some(ref loc) = self.loc {
            ifd.set_source_file(loc.filename.clone());
            ifd.with_primary_label(loc.line, loc.column, Some(self.message.clone()));
            if let Some(ref remark_name) = self.remark_name {
                ifd.with_note(format!("{} in pass {}", &remark_name, &self.pass_name));
            } else {
                ifd.with_note(format!("emitted from pass {}", &self.pass_name));
            }
        } else {
            ifd.with_note(self.message.clone());
            ifd.with_note(format!("for function {}", self.function_name()));
            if let Some(ref remark_name) = self.remark_name {
                ifd.with_note(format!("{} in pass {}", &remark_name, &self.pass_name));
            } else {
                ifd.with_note(format!("emitted from pass {}", &self.pass_name));
            }
        }

        if ifd.verbose() {
            if let Some(code_region) = self.code_region {
                let ir_ptr = unsafe { LLVMPrintValueToString(code_region) };
                let ir_cstr = unsafe { CStr::from_ptr(ir_ptr) };
                let ir = ir_cstr.to_string_lossy();
                ifd.with_note(format!("in reference to the following llvm ir: {}", ir));
            }
        }

        true
    }

    fn function_name(&self) -> &str {
        use llvm_sys::core::LLVMGetValueName2;
        unsafe {
            let mut len = MaybeUninit::<libc::size_t>::uninit();
            let ptr = LLVMGetValueName2(self.function, len.as_mut_ptr());
            let len = len.assume_init();
            let bytes = slice::from_raw_parts(ptr as *const u8, len as usize);
            std::str::from_utf8(bytes).unwrap()
        }
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

    fn format(&self, ifd: &mut InFlightDiagnostic, _context: &Context, options: &Options) -> bool {
        use llvm_sys::core::LLVMGetValueName2;

        if !options.debugging_opts.print_llvm_optimization_remarks {
            return false;
        }

        let fun = unsafe {
            let mut len = MaybeUninit::<libc::size_t>::uninit();
            let ptr = LLVMGetValueName2(self.function, len.as_mut_ptr());
            let len = len.assume_init();
            let bytes = slice::from_raw_parts(ptr as *const u8, len as usize);
            std::str::from_utf8(bytes).unwrap()
        };
        ifd.with_message(format!(
            "instruction selection used fallback path in {}",
            fun
        ));

        true
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
        use DiagnosticKind::*;

        let kind = LLVMLumenGetDiagInfoKind(di);
        match kind {
            OptimizationRemark
            | OptimizationRemarkMissed
            | OptimizationRemarkAnalysis
            | OptimizationRemarkAnalysisAliasing
            | OptimizationRemarkAnalysisFPCommute
            | OptimizationFailure
            | MachineOptimizationRemark
            | MachineOptimizationRemarkMissed
            | MachineOptimizationRemarkAnalysis => {
                Self::Optimization(OptimizationDiagnostic::unpack(kind.into(), di))
            }
            ISelFallback => Self::ISelFallback(ISelFallbackDiagnostic::unpack(di)),
            Linker => Self::Linker(di),
            _ => Self::Unknown(di),
        }
    }

    fn format(&self, ifd: &mut InFlightDiagnostic, context: &Context, options: &Options) -> bool {
        match self {
            Self::Optimization(info) => info.format(ifd, context, options),
            Self::ISelFallback(info) => info.format(ifd, context, options),
            Self::Linker(info) if ifd.severity() == Severity::Error => {
                ifd.with_message("cannot link module");
                let message = strings::build_string(|message| unsafe {
                    LLVMLumenWriteDiagnosticInfoToString(info, message);
                })
                .expect("expected diagnostic message, but got an empty string");
                ifd.with_note(message);
                true
            }
            Self::Linker(info) => {
                let message = strings::build_string(|message| unsafe {
                    LLVMLumenWriteDiagnosticInfoToString(info, message);
                })
                .expect("expected diagnostic message, but got an empty string");
                ifd.with_message(message);
                true
            }
            Self::Unknown(info) => {
                let message = strings::build_string(|message| unsafe {
                    LLVMLumenWriteDiagnosticInfoToString(info, message);
                })
                .expect("expected diagnostic message, but got an empty string");
                ifd.with_message(message);
                true
            }
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

    let (context, options, diagnostics) =
        &*(user as *const (&Context, Weak<Options>, Weak<DiagnosticsHandler>));
    if let Some(options) = options.upgrade() {
        if let Some(diagnostics) = diagnostics.upgrade() {
            let info = DiagnosticInfo::unpack(info);

            info.report(context, &options, &diagnostics);
        }
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
        remark_name_out: &RustString,
        function_out: *mut Value,
        code_region_out: *mut Value,
        loc_line_out: &mut libc::c_uint,
        loc_column_out: &mut libc::c_uint,
        is_verbose: &mut bool,
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
