use std::ffi::{CStr, CString};
use std::os;
use std::str::FromStr;
use std::sync::Arc;

use llvm_sys::target_machine::LLVMCodeGenFileType;

use liblumen_llvm::diagnostics::last_error;
use liblumen_session::{DiagnosticsHandler, OptLevel, Options, ProjectType};
use liblumen_target::{CodeModel, RelocMode, ThreadLocalMode};
use liblumen_util::error::FatalError;

use crate::ffi::{self, util};
use crate::llvm::{ModuleRef, TargetMachine, TargetMachineRef};

extern "C" {
    pub fn PrintTargetCPUs(TM: TargetMachineRef);

    pub fn PrintTargetFeatures(TM: TargetMachineRef);

    pub fn LLVMLumenCreateTargetMachine(
        triple: *const libc::c_char,
        cpu: *const libc::c_char,
        features: *const libc::c_char,
        abi: *const libc::c_char,
        code_model: CodeModel,
        reloc_mode: RelocMode,
        opt_level: ffi::CodeGenOptLevel,
        pic: bool,
        function_sections: bool,
        data_sections: bool,
        trap_unreachable: bool,
        single_thread: bool,
        asm_comments: bool,
        emit_stack_size_section: bool,
        relax_elf_relocations: bool,
    ) -> TargetMachineRef;

    pub fn LLVMLumenDisposeTargetMachine(T: TargetMachineRef);

    #[cfg(not(windows))]
    pub fn LLVMTargetMachineEmitToFileDescriptor(
        T: TargetMachineRef,
        M: ModuleRef,
        fd: os::unix::io::RawFd,
        codegen: LLVMCodeGenFileType,
        error_message: *mut *mut libc::c_char,
    ) -> bool;

    #[cfg(windows)]
    pub fn LLVMTargetMachineEmitToFileDescriptor(
        T: TargetMachineRef,
        M: ModuleRef,
        fd: os::windows::io::RawHandle,
        codegen: LLVMCodeGenFileType,
        error_message: *mut *mut libc::c_char,
    ) -> bool;
}

pub fn print_target_cpus(options: &Options, diagnostics: &DiagnosticsHandler) {
    util::require_inited();
    let tm = create_informational_target_machine(options, diagnostics, true);
    unsafe { PrintTargetCPUs(tm.as_ref()) };
}

pub fn print_target_features(options: &Options, diagnostics: &DiagnosticsHandler) {
    util::require_inited();
    let tm = create_informational_target_machine(options, diagnostics, true);
    unsafe { PrintTargetFeatures(tm.as_ref()) };
}

pub fn create_informational_target_machine(
    options: &Options,
    diagnostics: &DiagnosticsHandler,
    find_features: bool,
) -> TargetMachine {
    target_machine_factory(options, OptLevel::No, find_features)()
        .unwrap_or_else(|err| llvm_err(diagnostics, &err).raise())
}

pub fn create_target_machine(
    options: &Options,
    diagnostics: &DiagnosticsHandler,
    find_features: bool,
) -> TargetMachine {
    let opt_level = options.codegen_opts.opt_level.unwrap_or(OptLevel::No);
    target_machine_factory(options, opt_level, find_features)()
        .unwrap_or_else(|err| llvm_err(diagnostics, &err).raise())
}

fn target_machine_factory(
    options: &Options,
    opt_level: OptLevel,
    find_features: bool,
) -> Arc<dyn Fn() -> Result<TargetMachine, String> + Send + Sync> {
    let reloc_mode = get_reloc_mode(options);

    let (opt_level, _) = util::to_llvm_opt_settings(opt_level);

    let ffunction_sections = options.target.options.function_sections;
    let fdata_sections = ffunction_sections;

    let code_model_arg = options.codegen_opts.code_model.as_ref().or(options
        .target
        .options
        .code_model
        .as_ref());

    let code_model = code_model_arg
        .map(|s| CodeModel::from_str(s).expect("expected a valid CodeModel value"))
        .unwrap_or(CodeModel::None);

    let features = llvm_target_features(options).collect::<Vec<_>>();
    let mut singlethread = options.target.options.singlethread;

    // On the wasm target once the `atomics` feature is enabled that means that
    // we're no longer single-threaded, or otherwise we don't want LLVM to
    // lower atomic operations to single-threaded operations.
    if singlethread
        && options.target.llvm_target.contains("wasm32")
        && features.iter().any(|s| *s == "+atomics")
    {
        singlethread = false;
    }

    let triple = CString::new(options.target.llvm_target.as_str()).unwrap();
    let cpu = CString::new(target_cpu(options)).unwrap();
    let features = features.join(",");
    let features = CString::new(features).unwrap();
    let abi = CString::new(options.target.options.llvm_abiname.as_str()).unwrap();
    let is_pie_binary = !find_features && is_pie_binary(options);
    let trap_unreachable = options.target.options.trap_unreachable;
    let emit_stack_size_section = options.debugging_opts.emit_stack_sizes;

    let asm_comments = options.debugging_opts.asm_comments;
    let relax_elf_relocations = options.target.options.relax_elf_relocations;

    Arc::new(move || {
        let tm = unsafe {
            LLVMLumenCreateTargetMachine(
                triple.as_ptr(),
                cpu.as_ptr(),
                features.as_ptr(),
                abi.as_ptr(),
                code_model,
                reloc_mode,
                opt_level,
                is_pie_binary,
                ffunction_sections,
                fdata_sections,
                trap_unreachable,
                singlethread,
                asm_comments,
                emit_stack_size_section,
                relax_elf_relocations,
            )
        };

        if tm.is_null() {
            return Err(format!(
                "Could not create LLVM TargetMachine for triple: {}",
                triple.to_str().unwrap()
            ));
        }

        Ok(TargetMachine::new(tm))
    })
}

pub fn llvm_target_features(options: &Options) -> impl Iterator<Item = &str> {
    const LUMEN_SPECIFIC_FEATURES: &[&str] = &["crt-static"];

    let cmdline = options
        .codegen_opts
        .target_features
        .as_ref()
        .map(|s| s.as_str())
        .unwrap_or("")
        .split(',')
        .filter(|f| !LUMEN_SPECIFIC_FEATURES.iter().any(|s| f.contains(s)));
    options
        .target
        .options
        .features
        .split(',')
        .chain(cmdline)
        .filter(|l| l.is_empty())
}

pub fn target_cpu(options: &Options) -> &str {
    let name = match options.codegen_opts.target_cpu {
        Some(ref s) => &**s,
        None => &*options.target.options.cpu,
    };

    if name != "native" {
        return name;
    }

    unsafe {
        let cstr = CStr::from_ptr(llvm_sys::target_machine::LLVMGetHostCPUName());
        cstr.to_str().unwrap()
    }
}

pub fn is_pie_binary(options: &Options) -> bool {
    !is_any_library(options) && get_reloc_mode(options) == RelocMode::PIC
}

pub fn is_any_library(options: &Options) -> bool {
    options.project_type != ProjectType::Executable
}

pub fn get_reloc_mode(options: &Options) -> RelocMode {
    let arg = match options.codegen_opts.relocation_mode {
        Some(ref s) => &s[..],
        None => &options.target.options.relocation_model[..],
    };

    RelocMode::from_str(arg).unwrap()
}

pub fn get_tls_mode(options: &Options) -> ThreadLocalMode {
    let arg = match options.codegen_opts.tls_mode {
        Some(ref s) => &s[..],
        None => &options.target.options.tls_model[..],
    };

    ThreadLocalMode::from_str(arg).unwrap()
}

pub fn llvm_err(handler: &DiagnosticsHandler, msg: &str) -> FatalError {
    match last_error() {
        Some(err) => handler.fatal_str(&format!("{}: {}", msg, err)),
        None => handler.fatal_str(&msg),
    }
}
