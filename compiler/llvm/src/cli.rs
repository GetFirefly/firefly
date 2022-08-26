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
        .map(|arg| CString::new(arg.as_str()).unwrap())
        .collect::<Vec<_>>();

    // MLIR
    if options.debugging_opts.time_mlir_passes {
        args.push(CString::new("-mlir-timing").unwrap());
        args.push(CString::new("-mlir-timing-display=tree").unwrap());
    }
    if options.debugging_opts.mlir_print_passes_before {
        args.push(CString::new("-mlir-print-ir-before-all").unwrap());
        args.push(CString::new("-print-before-all").unwrap());
    }
    if options.debugging_opts.mlir_print_passes_after {
        args.push(CString::new("-mlir-print-ir-after-all").unwrap());
        args.push(CString::new("-print-after-all").unwrap());
    }
    if options.debugging_opts.mlir_print_passes_on_change {
        args.push(CString::new("-mlir-print-ir-after-change").unwrap());
        args.push(CString::new("-print-changed").unwrap());
    }
    if options.debugging_opts.mlir_print_passes_on_failure {
        args.push(CString::new("-mlir-print-ir-after-failure").unwrap());
    }
    if options.debugging_opts.mlir_print_module_scope {
        args.push(CString::new("-mlir-print-ir-module-scope").unwrap());
        args.push(CString::new("-print-module-scope").unwrap());
    }
    if options.debugging_opts.mlir_print_local_scope {
        args.push(CString::new("-mlir-print-local-scope").unwrap());
    }
    if options.debugging_opts.mlir_enable_statistics {
        args.push(CString::new("-mlir-pass-statistics").unwrap());
    }
    if options.debugging_opts.mlir_print_op_on_diagnostic {
        args.push(CString::new("-mlir-print-op-on-diagnostic").unwrap());
    }
    if options.debugging_opts.mlir_print_trace_on_diagnostic {
        args.push(CString::new("-mlir-print-stacktrace-on-diagnostic").unwrap());
    }
    if let Some(path) = options
        .debugging_opts
        .mlir_enable_crash_reproducer
        .as_deref()
    {
        args.push(
            CString::new(format!(
                "-mlir-pass-pipeline-crash-reproducer={}",
                path.display()
            ))
            .unwrap(),
        );
    }
    match options.debugging_opts.mlir_print_debug_info {
        MlirDebugPrinting::Plain => {
            args.push(CString::new("-mlir-print-debuginfo").unwrap());
        }
        MlirDebugPrinting::Pretty => {
            args.push(CString::new("-mlir-print-debuginfo").unwrap());
            args.push(CString::new("-mlir-pretty-debuginfo").unwrap());
        }
        _ => (),
    }
    if options.debugging_opts.mlir_print_generic_ops {
        args.push(CString::new("-mlir-print-op-generic").unwrap());
    } else {
        args.push(CString::new("-mlir-print-assume-verified").unwrap());
    }

    // LLVM
    if options.debugging_opts.time_llvm_passes {
        args.push(CString::new("-time-passes").unwrap());
    }
    match options
        .codegen_opts
        .merge_functions
        .unwrap_or(options.target.options.merge_functions)
    {
        MergeFunctions::Disabled | MergeFunctions::Trampolines => {}
        MergeFunctions::Aliases => {
            args.push(CString::new("-mergefunc-use-aliases").unwrap());
        }
    }

    // HACK: LLVM inserts `llvm.assume` calls to preserve align attributes
    // during inlining. Unfortunately these may block other optimizations.
    args.push(CString::new("-preserve-alignment-assumptions-during-inlining=false").unwrap());

    let ptrs = args.iter().map(|cstr| cstr.as_ptr()).collect::<Vec<_>>();
    let overview = std::ptr::null();

    unsafe {
        LLVMParseCommandLineOptions(ptrs.len().try_into().unwrap(), ptrs.as_ptr(), overview);
    }
}
