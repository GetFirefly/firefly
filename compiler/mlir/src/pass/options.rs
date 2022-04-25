use std::path::{Path, PathBuf};

use liblumen_llvm as llvm;
use liblumen_llvm::codegen::{CodeGenOptLevel, CodeGenOptSize};
use liblumen_session::{MlirDebugPrinting, Options};

#[repr(C)]
#[derive(Default)]
pub struct OpPrintingFlags {
    pub print_before_pass: bool,
    pub print_after_pass: bool,
    pub print_module_scope: bool,
    pub print_only_after_change: bool,
    pub print_only_after_failure: bool,
    pub enable_debug_info: bool,
    pub enable_pretty_debug_info: bool,
    pub print_generic_form: bool,
    pub use_local_scope: bool,
}
impl From<&PassManagerOptions> for OpPrintingFlags {
    fn from(opts: &PassManagerOptions) -> Self {
        Self {
            print_before_pass: opts.print_before_pass,
            print_after_pass: opts.print_after_pass,
            print_module_scope: opts.print_module_scope,
            print_only_after_change: opts.print_only_after_change,
            print_only_after_failure: opts.print_only_after_failure,
            enable_debug_info: opts.debug_info == MlirDebugPrinting::Plain,
            enable_pretty_debug_info: opts.debug_info == MlirDebugPrinting::Pretty,
            print_generic_form: opts.print_generic_form,
            use_local_scope: opts.print_local_scope,
        }
    }
}

/// This structure contains all of the options used to configure
/// the MLIR pass manager provided during construction.
#[derive(Debug, Default)]
pub struct PassManagerOptions {
    pub opt: CodeGenOptLevel,
    pub size_opt: CodeGenOptSize,
    pub enable_timing: bool,
    pub enable_statistics: bool,
    pub enable_verifier: bool,
    pub enable_crash_reproducer: Option<PathBuf>,
    pub debug_info: MlirDebugPrinting,
    pub print_generic_form: bool,
    pub print_before_pass: bool,
    pub print_after_pass: bool,
    /// NOTE: This should not be used when the pass manager is multi-threaded
    pub print_module_scope: bool,
    pub print_local_scope: bool,
    pub print_only_after_change: bool,
    pub print_only_after_failure: bool,
}
impl PassManagerOptions {
    pub fn new(options: &Options) -> Self {
        let (opt, size) = llvm::codegen::to_llvm_opt_settings(options.opt_level);
        Self {
            opt,
            size_opt: size,
            enable_timing: options.debugging_opts.mlir_enable_timing,
            enable_statistics: options.debugging_opts.mlir_enable_statistics,
            enable_verifier: options.debugging_opts.mlir_enable_verifier,
            enable_crash_reproducer: options.debugging_opts.mlir_enable_crash_reproducer.clone(),
            debug_info: options.debugging_opts.mlir_print_debug_info,
            print_generic_form: options.debugging_opts.mlir_print_generic_ops,
            print_before_pass: options.debugging_opts.mlir_print_passes_before,
            print_after_pass: options.debugging_opts.mlir_print_passes_after,
            print_module_scope: options.debugging_opts.mlir_print_module_scope,
            print_local_scope: options.debugging_opts.mlir_print_local_scope,
            print_only_after_change: options.debugging_opts.mlir_print_passes_on_change,
            print_only_after_failure: options.debugging_opts.mlir_print_passes_on_failure,
        }
    }

    pub fn printing_flags(&self) -> OpPrintingFlags {
        OpPrintingFlags::from(self)
    }

    #[inline(always)]
    pub fn opt_level(&self) -> CodeGenOptLevel {
        self.opt
    }

    #[inline(always)]
    pub fn size_opt_level(&self) -> CodeGenOptSize {
        self.size_opt
    }

    /// Returns the path to the crash reproducer, if enabled
    pub fn crash_reproducer(&self) -> Option<&Path> {
        self.enable_crash_reproducer.as_ref().map(|pb| pb.as_path())
    }
}
