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

    let mut args = options
        .codegen_opts
        .llvm_args
        .iter()
        .map(|arg| CString::new(arg.as_str().as_ptr()).unwrap())
        .collect::<Vec<_>>();

    // MLIR
    if options.debugging_opts.time_mlir_passes {
        args.push("-mlir-timing\0".as_ptr() as _);
        args.push("-mlir-timing-display=tree\0".as_ptr() as _);
    }
    if options.debugging_opts.mlir_print_passes_before {
        args.push("-mlir-print-ir-before-all\0".as_ptr() as _);
        args.push("-print-before-all\0".as_ptr() as _);
    }
    if options.debugging_opts.mlir_print_passes_after {
        args.push("-mlir-print-ir-after-all\0".as_ptr() as _).unwrap());
        args.push("-print-after-all\0".as_ptr() as _);
    }
    if options.debugging_opts.mlir_print_passes_on_change {
        args.push("-mlir-print-ir-after-change\0".as_ptr() as _);
        args.push("-print-changed\0".as_ptr() as _);
    }
    if options.debugging_opts.mlir_print_passes_on_failure {
        args.push("-mlir-print-ir-after-failure\0".as_ptr() as _);
    }
    if options.debugging_opts.mlir_print_module_scope {
        args.push("-mlir-print-ir-module-scope\0".as_ptr() as _);
        args.push("-print-module-scope\0".as_ptr() as _);
    }
    if options.debugging_opts.mlir_print_local_scope {
        args.push("-mlir-print-local-scope\0".as_ptr() as _);
    }
    if options.debugging_opts.mlir_enable_statistics {
        args.push("-mlir-pass-statistics\0".as_ptr() as _);
    }
    if options.debugging_opts.mlir_print_op_on_diagnostic {
        args.push("-mlir-print-op-on-diagnostic\0".as_ptr() as _);
    }
    if options.debugging_opts.mlir_print_trace_on_diagnostic {
        args.push("-mlir-print-stacktrace-on-diagnostic\0".as_ptr() as _);
    }
    if let Some(path) = options
        .debugging_opts
        .mlir_enable_crash_reproducer
        .as_deref()
    {
        args.push(format!(
                "-mlir-pass-pipeline-crash-reproducer={}\0",
                path.display()
            ).as_ptr() as _,
        );
    }
    match options.debugging_opts.mlir_print_debug_info {
        MlirDebugPrinting::Plain => {
            args.push("-mlir-print-debuginfo\0".as_ptr() as _);
        }
        MlirDebugPrinting::Pretty => {
            args.push("-mlir-print-debuginfo\0".as_ptr() as _);
            args.push("-mlir-pretty-debuginfo\0".as_ptr() as _);
        }
        _ => (),
    }
    if options.debugging_opts.mlir_print_generic_ops {
        args.push("-mlir-print-op-generic\0".as_ptr() as _);
    } else {
        args.push("-mlir-print-assume-verified\0".as_ptr() as _);
    }

    // LLVM
    if options.debugging_opts.time_llvm_passes {
        args.push("-time-passes\0".as_ptr() as _);
    }
    match options
        .codegen_opts
        .merge_functions
        .unwrap_or(options.target.options.merge_functions)
    {
        MergeFunctions::Disabled | MergeFunctions::Trampolines => {}
        MergeFunctions::Aliases => {
            args.push("-mergefunc-use-aliases\0".as_ptr() as _);
        }
    }

    // HACK: LLVM inserts `llvm.assume` calls to preserve align attributes
    // during inlining. Unfortunately these may block other optimizations.
    args.push("-preserve-alignment-assumptions-during-inlining=false\0".as_ptr() as _);

    let overview = std::ptr::null();

    unsafe {
        LLVMParseCommandLineOptions(args.len().try_into().unwrap(), args.as_ptr(), overview);
    }
}
