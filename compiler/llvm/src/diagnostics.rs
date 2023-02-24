use std::fmt;
use std::mem::MaybeUninit;

use firefly_session::Options;
use firefly_util as util;
use firefly_util::diagnostics::{FileName, InFlightDiagnostic, Severity};

use crate::ir::{Context, Function, Value, ValueBase};
use crate::support::*;

extern "C" {
    type LlvmDiagnostic;
}

pub type DiagnosticHandler = unsafe extern "C" fn(DiagnosticBase, *mut core::ffi::c_void);

/// Initializes the diagnostic system for use
pub fn init() {
    extern "C" {
        fn LLVMFireflyInstallFatalErrorHandler();
    }
    unsafe {
        LLVMFireflyInstallFatalErrorHandler();
    }
}

/// Represents a diagnostic's severity in LLVM
///
/// NOTE: We decorate the variants with allow(dead_code) because
/// they are never constructed in Rust, only in C++. We also only
/// derive impls for features we actually use on this enum currently
#[repr(C)]
#[derive(PartialEq, Eq)]
enum DiagnosticInfoSeverity {
    #[allow(dead_code)]
    Error = 0,
    #[allow(dead_code)]
    Warning,
    #[allow(dead_code)]
    Remark,
    #[allow(dead_code)]
    Note,
}
impl Into<Severity> for DiagnosticInfoSeverity {
    fn into(self) -> Severity {
        match self {
            Self::Error => Severity::Error,
            Self::Warning => Severity::Warning,
            Self::Remark | Self::Note => Severity::Note,
        }
    }
}

/// The kind of a diagnostic
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(C)]
pub enum DiagnosticKind {
    InlineAsm = 0,
    ResourceLimit,
    StackSize,
    Linker,
    Lowering,
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
    Unsupported,
    SourceManager,
    DontCall,
    Other,
}
impl DiagnosticKind {
    pub fn describe(self) -> &'static str {
        match self {
            Self::InlineAsm => "inline asm",
            Self::ResourceLimit => "resource limit",
            Self::StackSize => "stack size",
            Self::Linker => "linker",
            Self::Lowering => "lowering",
            Self::DebugMetadataVersion => "debug metadata version",
            Self::DebugMetadataInvalid => "invalid debug metadata",
            Self::ISelFallback => "instruction selection fallback",
            Self::OptimizationRemark | Self::MachineOptimizationRemark => "optimization remark",
            Self::OptimizationRemarkMissed | Self::MachineOptimizationRemarkMissed => {
                "optimization missed"
            }
            Self::OptimizationRemarkAnalysis | Self::MachineOptimizationRemarkAnalysis => {
                "optimization analysis"
            }
            Self::OptimizationRemarkAnalysisFPCommute => {
                "optimization analysis (floating-point commute)"
            }
            Self::OptimizationRemarkAnalysisAliasing => "optimization analysis (aliasing)",
            Self::OptimizationFailure => "optimization failure",
            Self::SampleProfile => "sample profiler",
            Self::MIRParser => "machine ir parser",
            Self::PGOProfile => "profile-guided optimization profiler",
            Self::Unsupported => "unsupported feature",
            Self::SourceManager => "source diagnostic",
            Self::DontCall => "dontcall",
            Self::Other => "other",
        }
    }

    pub fn is_optimization_remark(self) -> bool {
        match self {
            Self::OptimizationRemark
            | Self::OptimizationRemarkMissed
            | Self::OptimizationRemarkAnalysis
            | Self::OptimizationRemarkAnalysisFPCommute
            | Self::OptimizationRemarkAnalysisAliasing
            | Self::OptimizationFailure
            | Self::MachineOptimizationRemark
            | Self::MachineOptimizationRemarkMissed
            | Self::MachineOptimizationRemarkAnalysis => true,
            _ => false,
        }
    }
}

/// This trait is implemented for all diagnostic types
pub trait Diagnostic {
    /// Return a string describing this diagnostic
    fn description(&self) -> OwnedStringRef {
        extern "C" {
            fn LLVMGetDiagInfoDescription(d: DiagnosticBase) -> *mut std::os::raw::c_char;
        }
        unsafe { OwnedStringRef::from_ptr(LLVMGetDiagInfoDescription(self.base())) }
    }

    /// Return the severity of this diagnostic
    fn severity(&self) -> Severity {
        extern "C" {
            fn LLVMGetDiagInfoSeverity(di: DiagnosticBase) -> DiagnosticInfoSeverity;
        }
        unsafe { LLVMGetDiagInfoSeverity(self.base()) }.into()
    }

    /// Return the kind of diagnostic this is
    fn kind(&self) -> DiagnosticKind {
        extern "C" {
            fn LLVMFireflyGetDiagInfoKind(info: DiagnosticBase) -> DiagnosticKind;
        }
        unsafe { LLVMFireflyGetDiagInfoKind(self.base()) }
    }

    fn report(
        &self,
        context: Context,
        options: &Options,
        emitter: &util::diagnostics::DiagnosticsHandler,
    ) {
        let severity = self.severity();
        let kind = self.kind();

        // If optimization remarks weren't requested, don't bother with them at all
        if kind.is_optimization_remark() && !options.debugging_opts.print_llvm_optimization_remarks
        {
            return;
        }

        let ifd = emitter.diagnostic(severity);
        self.format(ifd, context, options);
    }

    /// Formats this diagnostic into the given InFlightDiagnostic
    ///
    /// Returns true if this diagnostic should be emitted, otherwise it should be dropped
    fn format(&self, ifd: InFlightDiagnostic, context: Context, options: &Options);

    fn base(&self) -> DiagnosticBase;
}

/// The base type for all diagnostics
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct DiagnosticBase(*const LlvmDiagnostic);
impl Diagnostic for DiagnosticBase {
    #[inline(always)]
    fn base(&self) -> Self {
        *self
    }

    fn format(&self, ifd: InFlightDiagnostic, _context: Context, _options: &Options) {
        ifd.with_message(self.description()).emit();
    }
}
impl fmt::Display for DiagnosticBase {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let description = self.description();
        write!(f, "{}", &description)
    }
}

/// Represents diagnostics related to optimization remarks, analyses, or errors
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct OptimizationDiagnostic(DiagnosticBase);
impl OptimizationDiagnostic {
    fn is_verbose(&self) -> bool {
        extern "C" {
            fn LLVMFireflyOptimizationDiagnosticIsVerbose(d: OptimizationDiagnostic) -> bool;
        }
        unsafe { LLVMFireflyOptimizationDiagnosticIsVerbose(*self) }
    }

    fn pass_name(&self) -> StringRef {
        extern "C" {
            fn LLVMFireflyOptimizationDiagnosticPassName(d: OptimizationDiagnostic) -> StringRef;
        }
        unsafe { LLVMFireflyOptimizationDiagnosticPassName(*self) }
    }

    fn remark_name(&self) -> StringRef {
        extern "C" {
            fn LLVMFireflyOptimizationDiagnosticRemarkName(d: OptimizationDiagnostic) -> StringRef;
        }
        unsafe { LLVMFireflyOptimizationDiagnosticRemarkName(*self) }
    }

    fn message(&self) -> OwnedStringRef {
        extern "C" {
            fn LLVMFireflyOptimizationDiagnosticMessage(
                d: OptimizationDiagnostic,
            ) -> *const std::os::raw::c_char;
        }
        unsafe { OwnedStringRef::from_ptr(LLVMFireflyOptimizationDiagnosticMessage(*self)) }
    }

    fn code_region(&self) -> Option<ValueBase> {
        extern "C" {
            fn LLVMFireflyOptimizationDiagnosticCodeRegion(d: OptimizationDiagnostic) -> ValueBase;
        }
        let value = unsafe { LLVMFireflyOptimizationDiagnosticCodeRegion(*self) };
        if value.is_null() {
            None
        } else {
            Some(value)
        }
    }

    fn function(&self) -> Function {
        extern "C" {
            fn LLVMFireflyDiagnosticWithLocFunction(d: DiagnosticBase) -> Function;
        }
        let function = unsafe { LLVMFireflyDiagnosticWithLocFunction(self.0) };
        assert!(!function.is_null());
        function
    }

    fn source_location(&self) -> Option<SourceLoc> {
        extern "C" {
            fn LLVMFireflyDiagnosticWithLocSourceLoc(
                d: DiagnosticBase,
                path: *mut StringRef,
                line: *mut u32,
                col: *mut u32,
            ) -> bool;
        }
        let mut path = MaybeUninit::uninit();
        let mut line = MaybeUninit::uninit();
        let mut col = MaybeUninit::uninit();
        let valid = unsafe {
            LLVMFireflyDiagnosticWithLocSourceLoc(
                self.0,
                path.as_mut_ptr(),
                line.as_mut_ptr(),
                col.as_mut_ptr(),
            )
        };
        if valid {
            Some(unsafe {
                SourceLoc::new(path.assume_init(), line.assume_init(), col.assume_init())
            })
        } else {
            None
        }
    }
}
impl Diagnostic for OptimizationDiagnostic {
    #[inline]
    fn base(&self) -> DiagnosticBase {
        self.0
    }

    fn format(&self, ifd: InFlightDiagnostic, _context: Context, options: &Options) {
        // Don't print noisy remarks
        if self.is_verbose() {
            return;
        }

        // Don't print optimization diagnostics other than failures, unless explicitly requested
        let kind = self.kind();
        if !options.debugging_opts.print_llvm_optimization_remarks {
            if kind != DiagnosticKind::OptimizationFailure {
                return;
            }
        }

        let pass_name = self.pass_name();
        let remark_name = self.remark_name();
        let message = self.message().to_string();
        let ifd = if let Some(loc) = self.source_location() {
            ifd.set_source_file(loc.filename)
                .with_message(kind.describe())
                .with_primary_label_line_and_col(loc.line, loc.column, Some(message))
                .with_note(format!("{} in pass {}", &remark_name, &pass_name))
        } else {
            let function = self.function();
            ifd.with_note(message)
                .with_note(format!("for function {}", function.name()))
                .with_note(format!("{} in pass {}", &remark_name, &pass_name))
        };

        if ifd.verbose() {
            if let Some(code_region) = self.code_region() {
                let code = code_region.to_string();
                ifd.with_note(format!("in reference to the following llvm ir: {}", &code))
                    .emit();
            } else {
                ifd.emit();
            }
        } else {
            ifd.emit();
        }
    }
}

/// Represents diagnostic info related to the linker
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct LinkerDiagnostic(DiagnosticBase);
impl Diagnostic for LinkerDiagnostic {
    #[inline]
    fn base(&self) -> DiagnosticBase {
        self.0
    }

    fn format(&self, ifd: InFlightDiagnostic, _context: Context, _options: &Options) {
        if ifd.severity() == Severity::Error {
            ifd.with_message("cannot link module")
                .with_note(self.description())
                .emit();
        } else {
            ifd.with_message(self.description()).emit();
        }
    }
}

/// Represents diagnostics generated when instruction selection fallback occurs
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ISelFallbackDiagnostic(DiagnosticBase);
impl ISelFallbackDiagnostic {
    fn function(&self) -> Function {
        extern "C" {
            fn LLVMFireflyISelFallbackDiagnosticFunction(d: DiagnosticBase) -> Function;
        }
        unsafe { LLVMFireflyISelFallbackDiagnosticFunction(self.0) }
    }
}
impl Diagnostic for ISelFallbackDiagnostic {
    #[inline]
    fn base(&self) -> DiagnosticBase {
        self.0
    }

    fn format(&self, ifd: InFlightDiagnostic, _context: Context, options: &Options) {
        if !options.debugging_opts.print_llvm_optimization_remarks {
            return;
        }

        let function = self.function();
        ifd.with_message(format!(
            "instruction selection used fallback path in {}",
            function.name()
        ))
        .emit();
    }
}

/// Represents the source location to which a diagnostic applies
pub struct SourceLoc {
    filename: FileName,
    line: u32,
    column: u32,
}
impl SourceLoc {
    fn new(filename: StringRef, line: u32, column: u32) -> Self {
        let path = filename.to_path_lossy();
        if path.exists() {
            Self {
                filename: FileName::from(path.into_owned()),
                line,
                column,
            }
        } else {
            Self {
                filename: FileName::from(filename.to_string()),
                line,
                column,
            }
        }
    }
}

pub(crate) unsafe extern "C" fn diagnostic_handler(
    diagnostic: DiagnosticBase,
    user: *mut core::ffi::c_void,
) {
    use std::sync::Weak;

    if user.is_null() {
        return;
    }

    let (context, options, diagnostics) = &*(user as *const (
        Context,
        Weak<Options>,
        Weak<util::diagnostics::DiagnosticsHandler>,
    ));
    if let Some(options) = options.upgrade() {
        if let Some(diagnostics) = diagnostics.upgrade() {
            let kind = diagnostic.kind();
            match kind {
                DiagnosticKind::OptimizationRemark
                | DiagnosticKind::OptimizationRemarkMissed
                | DiagnosticKind::OptimizationRemarkAnalysis
                | DiagnosticKind::OptimizationRemarkAnalysisFPCommute
                | DiagnosticKind::OptimizationRemarkAnalysisAliasing
                | DiagnosticKind::OptimizationFailure
                | DiagnosticKind::MachineOptimizationRemark
                | DiagnosticKind::MachineOptimizationRemarkMissed
                | DiagnosticKind::MachineOptimizationRemarkAnalysis => {
                    let di = OptimizationDiagnostic(diagnostic);
                    di.report(*context, &options, &diagnostics);
                }
                DiagnosticKind::Linker => {
                    let di = LinkerDiagnostic(diagnostic);
                    di.report(*context, &options, &diagnostics);
                }
                DiagnosticKind::ISelFallback => {
                    let di = ISelFallbackDiagnostic(diagnostic);
                    di.report(*context, &options, &diagnostics);
                }
                // We don't have any special handling for these kinds currently
                DiagnosticKind::InlineAsm
                | DiagnosticKind::ResourceLimit
                | DiagnosticKind::StackSize
                | DiagnosticKind::Lowering
                | DiagnosticKind::DebugMetadataVersion
                | DiagnosticKind::DebugMetadataInvalid
                | DiagnosticKind::SampleProfile
                | DiagnosticKind::MIRParser
                | DiagnosticKind::PGOProfile
                | DiagnosticKind::Unsupported
                | DiagnosticKind::SourceManager
                | DiagnosticKind::DontCall
                | DiagnosticKind::Other => {
                    diagnostic.report(*context, &options, &diagnostics);
                }
            }
        }
    }
}
