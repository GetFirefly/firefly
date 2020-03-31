use std::ffi::CString;

use liblumen_session::Options;
use liblumen_target::MergeFunctions;

pub fn init(options: &Options) {
    let mut args = options.codegen_opts.llvm_args.clone();

    if options.debugging_opts.time_llvm_passes {
        args.push("-time-passes".to_owned());
    }

    match options
        .debugging_opts
        .merge_functions
        .unwrap_or(options.target.options.merge_functions)
    {
        MergeFunctions::Disabled | MergeFunctions::Trampolines => {}
        MergeFunctions::Aliases => {
            args.push("-mergefunc-use-aliases".to_owned());
        }
    }

    // HACK: LLVM inserts `llvm.assume` calls to preserve align attributes
    // during inlining. Unfortunately these may block other optimizations.
    args.push("-preserve-alignment-assumptions-during-inlining=false".to_owned());

    // Pass configuration options to LLVM
    set_options(args.as_slice());
}

fn set_options(args: &[String]) {
    let n_args = args.len();
    let mut llvm_c_strs = Vec::with_capacity(n_args + 1);
    let mut llvm_args = Vec::with_capacity(n_args + 1);

    // Construct arguments for LLVM
    {
        let mut add = |arg: &str| {
            let s = CString::new(arg).unwrap();
            llvm_args.push(s.as_ptr());
            llvm_c_strs.push(s);
        };
        add("lumen"); // fake program name
        for arg in args {
            add(arg.as_str());
        }
    }

    unsafe {
        LLVMLumenSetLLVMOptions(llvm_args.len() as libc::c_int, llvm_args.as_ptr());
    }
}

extern "C" {
    pub fn LLVMLumenSetLLVMOptions(argc: libc::c_int, argv: *const *const libc::c_char);
}
