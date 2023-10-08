use std::ffi::CString;

use firefly_session::{MlirDebugPrinting, Options};
use firefly_target::MergeFunctions;

/// Configures LLVM/MLIR using passed-through command-line arguments, or generated from our own
/// compiler options that translate to known LLVM arguments.
///
/// NOTE: This only needs to be performed once on startup, and should be after mlir::init if
/// also configuring MLIR.
pub fn init(options: &Options) {
    extern "C" {
        fn LLVMParseCommandLineOptions(
            argc: i32,
            argv: *const *const std::os::raw::c_char,
            overview: *const std::os::raw::c_char,
        );
    }

    let args = options
        .codegen_opts
        .llvm_args
        .iter()
        .map(|arg| CString::new(arg.as_str()).unwrap())
        .collect::<Vec<_>>();

    let mut ptrs = args.iter().map(|cstr| cstr.as_ptr()).collect::<Vec<_>>();

    // MLIR
    if options.debugging_opts.time_mlir_passes {
        ptrs.push("-mlir-timing\0".as_ptr() as _);
        ptrs.push("-mlir-timing-display=tree\0".as_ptr() as _);
    }
    if options.debugging_opts.mlir_print_passes_before {
        ptrs.push("-mlir-print-ir-before-all\0".as_ptr() as _);
        ptrs.push("-print-before-all\0".as_ptr() as _);
    }
    if options.debugging_opts.mlir_print_passes_after {
        ptrs.push("-mlir-print-ir-after-all\0".as_ptr() as _);
        ptrs.push("-print-after-all\0".as_ptr() as _);
    }
    if options.debugging_opts.mlir_print_passes_on_change {
        ptrs.push("-mlir-print-ir-after-change\0".as_ptr() as _);
        ptrs.push("-print-changed\0".as_ptr() as _);
    }
    if options.debugging_opts.mlir_print_passes_on_failure {
        ptrs.push("-mlir-print-ir-after-failure\0".as_ptr() as _);
    }
    if options.debugging_opts.mlir_print_module_scope {
        ptrs.push("-mlir-print-ir-module-scope\0".as_ptr() as _);
        ptrs.push("-print-module-scope\0".as_ptr() as _);
    }
    if options.debugging_opts.mlir_print_local_scope {
        ptrs.push("-mlir-print-local-scope\0".as_ptr() as _);
    }
    if options.debugging_opts.mlir_enable_statistics {
        ptrs.push("-mlir-pass-statistics\0".as_ptr() as _);
    }
    if options.debugging_opts.mlir_print_op_on_diagnostic {
        ptrs.push("-mlir-print-op-on-diagnostic\0".as_ptr() as _);
    }
    if options.debugging_opts.mlir_print_trace_on_diagnostic {
        ptrs.push("-mlir-print-stacktrace-on-diagnostic\0".as_ptr() as _);
    }
    if let Some(path) = options
        .debugging_opts
        .mlir_enable_crash_reproducer
        .as_deref()
    {
        ptrs.push(format!(
                "-mlir-pass-pipeline-crash-reproducer={}\0",
                path.display()
            ).as_ptr() as _,
        );
    }
    match options.debugging_opts.mlir_print_debug_info {
        MlirDebugPrinting::Plain => {
            ptrs.push("-mlir-print-debuginfo\0".as_ptr() as _);
        }
        MlirDebugPrinting::Pretty => {
            ptrs.push("-mlir-print-debuginfo\0".as_ptr() as _);
            ptrs.push("-mlir-pretty-debuginfo\0".as_ptr() as _);
        }
        _ => (),
    }
    if options.debugging_opts.mlir_print_generic_ops {
        ptrs.push("-mlir-print-op-generic\0".as_ptr() as _);
    } else {
        ptrs.push("-mlir-print-assume-verified\0".as_ptr() as _);
    }

    // LLVM
    if options.debugging_opts.time_llvm_passes {
        ptrs.push("-time-passes\0".as_ptr() as _);
    }
    match options
        .codegen_opts
        .merge_functions
        .unwrap_or(options.target.options.merge_functions)
    {
        MergeFunctions::Disabled | MergeFunctions::Trampolines => {}
        MergeFunctions::Aliases => {
            ptrs.push("-mergefunc-use-aliases\0".as_ptr() as _);
        }
    }

    // HACK: LLVM inserts `llvm.assume` calls to preserve align attributes
    // during inlining. Unfortunately these may block other optimizations.
    ptrs.push("-preserve-alignment-assumptions-during-inlining=false\0".as_ptr() as _);

    let overview = std::ptr::null();

    unsafe {
        LLVMParseCommandLineOptions(ptrs.len().try_into().unwrap(), ptrs.as_ptr(), overview);
    }
}
