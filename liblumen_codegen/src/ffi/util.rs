use std::ffi::CString;
use std::sync::Once;

use liblumen_session::{OptLevel, Options};
use liblumen_target::{MergeFunctions, PanicStrategy};

use crate::ffi::{CodeGenOptLevel, CodeGenOptSize};

static INIT: Once = Once::new();

pub fn init(options: &Options) {
    unsafe {
        // Before we touch LLVM, make sure that multithreading is enabled.
        INIT.call_once(|| {
            assert_eq!(
                llvm_sys::core::LLVMIsMultithreaded(),
                1,
                "expected LLVM to be compiled with multithreading enabled"
            );
            configure_llvm(options);
        });
    }
}

pub(crate) fn require_inited() {
    INIT.call_once(|| panic!("LLVM is not initialized"));
}

unsafe fn configure_llvm(options: &Options) {
    let n_args = options.codegen_opts.llvm_args.len();
    let mut llvm_c_strs = Vec::with_capacity(n_args + 1);
    let mut llvm_args = Vec::with_capacity(n_args + 1);

    // Handle fatal errors better
    LLVMLumenInstallFatalErrorHandler();

    // Construct arguments for LLVM
    {
        let mut add = |arg: &str| {
            let s = CString::new(arg).unwrap();
            llvm_args.push(s.as_ptr());
            llvm_c_strs.push(s);
        };
        add("lumen"); // fake program name
        if options.debugging_opts.time_llvm_passes {
            add("-time-passes");
        }
        match options
            .debugging_opts
            .merge_functions
            .unwrap_or(options.target.options.merge_functions)
        {
            MergeFunctions::Disabled | MergeFunctions::Trampolines => {}
            MergeFunctions::Aliases => {
                add("-mergefunc-use-aliases");
            }
        }

        if options.target.target_os == "emscripten"
            && options.codegen_opts.panic.unwrap_or(PanicStrategy::Unwind) == PanicStrategy::Unwind
        {
            add("-enable-emscripten-cxx-exceptions");
        }

        // HACK(eddyb) LLVM inserts `llvm.assume` calls to preserve align attributes
        // during inlining. Unfortunately these may block other optimizations.
        add("-preserve-alignment-assumptions-during-inlining=false");

        for arg in &options.codegen_opts.llvm_args {
            add(&(*arg));
        }
    }

    // Initialize passes
    LLVMLumenInitializePasses();

    // Initialize all targets
    llvm_sys::target::LLVM_InitializeAllTargetInfos();
    llvm_sys::target::LLVM_InitializeAllTargets();
    llvm_sys::target::LLVM_InitializeAllTargetMCs();
    llvm_sys::target::LLVM_InitializeAllAsmPrinters();
    llvm_sys::target::LLVM_InitializeAllAsmParsers();
    llvm_sys::target::LLVM_InitializeAllDisassemblers();

    LLVMLumenSetLLVMOptions(llvm_args.len() as libc::c_int, llvm_args.as_ptr());
}

pub fn to_llvm_opt_settings(cfg: OptLevel) -> (CodeGenOptLevel, CodeGenOptSize) {
    match cfg {
        OptLevel::No => (CodeGenOptLevel::None, CodeGenOptSize::None),
        OptLevel::Less => (CodeGenOptLevel::Less, CodeGenOptSize::None),
        OptLevel::Default => (CodeGenOptLevel::Default, CodeGenOptSize::None),
        OptLevel::Aggressive => (CodeGenOptLevel::Aggressive, CodeGenOptSize::None),
        OptLevel::Size => (CodeGenOptLevel::Default, CodeGenOptSize::Default),
        OptLevel::SizeMin => (CodeGenOptLevel::Default, CodeGenOptSize::Aggressive),
    }
}

pub fn llvm_version() -> String {
    // Can be called without initializing LLVM
    unsafe { format!("{}.{}", LLVMLumenVersionMajor(), LLVMLumenVersionMinor()) }
}

extern "C" {
    pub fn LLVMLumenInstallFatalErrorHandler();
    pub fn LLVMLumenInitializePasses();
    pub fn LLVMLumenSetLLVMOptions(argc: libc::c_int, argv: *const *const libc::c_char);
    pub fn LLVMLumenVersionMajor() -> u32;
    pub fn LLVMLumenVersionMinor() -> u32;
}
