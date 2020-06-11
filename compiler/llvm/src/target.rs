use std::ffi::{CStr, CString};
use std::fmt;
use std::os;
use std::ptr;
use std::str::FromStr;

use anyhow::anyhow;

use liblumen_session::{OptLevel, Options, ProjectType};
use liblumen_target::{CodeModel, RelocModel, ThreadLocalMode};

use crate::enums::{self, CodeGenOptLevel, CodeGenOptSize};
use crate::module::ModuleRef;
use crate::sys as llvm_sys;
use crate::sys::target_machine::LLVMCodeGenFileType;

pub type TargetMachineRef = llvm_sys::target_machine::LLVMTargetMachineRef;
pub type TargetDataRef = llvm_sys::target::LLVMTargetDataRef;

mod ffi {
    use liblumen_target::{CodeModel, RelocModel};
    use crate::enums::{CodeGenOptLevel, CodeGenOptSize};

    #[repr(C)]
    pub(super) struct TargetFeature<'a> {
        name: *const u8,
        name_len: libc::c_uint,
        _marker: std::marker::PhantomData<&'a str>,
    }
    impl<'a> TargetFeature<'a> {
        pub(super) fn new(s: &str) -> Self {
            Self {
                name: s.as_ptr(),
                name_len: s.len() as libc::c_uint,
                _marker: std::marker::PhantomData,
            }
        }
    }

    #[repr(C)]
    pub(super) struct TargetMachineConfig<'a> {
        pub triple: *const u8,
        pub triple_len: libc::c_uint,
        pub cpu: *const u8,
        pub cpu_len: libc::c_uint,
        pub abi: *const u8,
        pub abi_len: libc::c_uint,
        pub features: *const TargetFeature<'a>,
        pub features_len: libc::c_uint,
        pub relax_elf_relocations: bool,
        pub position_independent_code: bool,
        pub data_sections: bool,
        pub function_sections: bool,
        pub emit_stack_size_section: bool,
        pub preserve_asm_comments: bool,
        pub enable_threading: bool,
        pub code_model: CodeModel,
        pub reloc_model: RelocModel,
        pub opt_level: CodeGenOptLevel,
        pub size_level: CodeGenOptSize,
    }
}

/// Initialize all targets
pub fn init() {
    unsafe {
        llvm_sys::target::LLVM_InitializeAllTargetInfos();
        llvm_sys::target::LLVM_InitializeAllTargets();
        llvm_sys::target::LLVM_InitializeAllTargetMCs();
        llvm_sys::target::LLVM_InitializeAllAsmPrinters();
        llvm_sys::target::LLVM_InitializeAllAsmParsers();
        llvm_sys::target::LLVM_InitializeAllDisassemblers();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetMachineConfig {
    triple: String,
    cpu: String,
    abi: String,
    features: Vec<String>,
    relax_elf_relocations: bool,
    position_independent_code: bool,
    data_sections: bool,
    function_sections: bool,
    emit_stack_size_section: bool,
    preserve_asm_comments: bool,
    enable_threading: bool,
    code_model: CodeModel,
    reloc_model: RelocModel,
    opt_level: CodeGenOptLevel,
    size_level: CodeGenOptSize,
}
impl TargetMachineConfig {
    pub fn new(options: &Options) -> Self {
        let reloc_model = get_reloc_mode(options);
        let (opt_level, size_level) = enums::to_llvm_opt_settings(options.opt_level);

        let code_model_arg = options.codegen_opts.code_model.as_ref().or(options
            .target
            .options
            .code_model
            .as_ref());

        let code_model = code_model_arg
            .map(|s| CodeModel::from_str(s).expect("expected a valid CodeModel value"))
            .unwrap_or(CodeModel::None);

        let features = llvm_target_features(options)
            .map(|f| f.to_owned())
            .collect::<Vec<_>>();

        let mut enable_threading = !options.target.options.singlethread;

        // On the wasm target once the `atomics` feature is enabled that means that
        // we're no longer single-threaded, or otherwise we don't want LLVM to
        // lower atomic operations to single-threaded operations.
        if !enable_threading
            && options.target.llvm_target.contains("wasm32")
            && features.iter().any(|s| s == "+atomics")
        {
            enable_threading = true;
        }

        let triple = options.target.llvm_target.clone();
        let cpu = target_cpu(options).to_string();
        let abi = options.target.options.llvm_abiname.clone();
        let is_pie_binary = is_pie_binary(options);
        let ffunction_sections = options.target.options.function_sections;

        Self {
            triple,
            cpu,
            abi,
            features,
            relax_elf_relocations: options.target.options.relax_elf_relocations,
            position_independent_code: is_pie_binary,
            data_sections: ffunction_sections,
            function_sections: ffunction_sections,
            emit_stack_size_section: options.debugging_opts.emit_stack_sizes,
            preserve_asm_comments: options.debugging_opts.asm_comments,
            enable_threading,
            code_model,
            reloc_model,
            opt_level,
            size_level,
        }
    }

    pub fn create(&self) -> anyhow::Result<TargetMachine> {
        crate::require_inited();

        let triple = self.triple.as_ptr();
        let cpu = self.cpu.as_ptr();
        let abi = self.abi.as_ptr();
        let features = self.features
            .iter()
            .map(|f| ffi::TargetFeature::new(f))
            .collect::<Vec<_>>();

        let config = ffi::TargetMachineConfig {
            triple,
            triple_len: self.triple.len() as libc::c_uint,
            cpu,
            cpu_len: self.cpu.len() as libc::c_uint,
            abi,
            abi_len: self.abi.len() as libc::c_uint,
            features: features.as_ptr(),
            features_len: features.len() as libc::c_uint,
            relax_elf_relocations: self.relax_elf_relocations,
            position_independent_code: self.position_independent_code,
            data_sections: self.data_sections,
            function_sections: self.function_sections,
            emit_stack_size_section: self.emit_stack_size_section,
            preserve_asm_comments: self.preserve_asm_comments,
            enable_threading: self.enable_threading,
            code_model: self.code_model,
            reloc_model: self.reloc_model,
            opt_level: self.opt_level,
            size_level: self.size_level,
        };

        let tm = unsafe { LLVMLumenCreateTargetMachine(&config) };
        if tm.is_null() {
            return Err(anyhow!(format!(
                "Could not create LLVM target machine for triple: {}",
                self.triple
            )));
        }

        Ok(TargetMachine::new(tm))
    }
}

#[repr(transparent)]
pub struct TargetMachine(TargetMachineRef);
impl TargetMachine {
    pub fn new(ptr: TargetMachineRef) -> Self {
        assert!(!ptr.is_null());
        Self(ptr)
    }

    pub fn as_ref(&self) -> TargetMachineRef {
        self.0
    }

    pub fn print_target_cpus(&self) {
        unsafe { PrintTargetCPUs(self.0) }
    }

    pub fn print_target_features(&self) {
        unsafe { PrintTargetFeatures(self.0) };
    }

    pub fn get_target_data(&self) -> TargetData {
        use llvm_sys::target_machine::LLVMCreateTargetDataLayout;
        let ptr = unsafe { LLVMCreateTargetDataLayout(self.0) };
        TargetData::new(ptr)
    }
}
unsafe impl Send for TargetMachine {}
unsafe impl Sync for TargetMachine {}
impl Eq for TargetMachine {}
impl PartialEq for TargetMachine {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.0, other.0)
    }
}
impl fmt::Debug for TargetMachine {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TargetMachine({:p})", self.0)
    }
}
impl Drop for TargetMachine {
    fn drop(&mut self) {
        unsafe {
            LLVMLumenDisposeTargetMachine(self.0);
        }
    }
}

#[repr(transparent)]
pub struct TargetData(TargetDataRef);
impl TargetData {
    pub fn new(ptr: TargetDataRef) -> Self {
        assert!(!ptr.is_null());
        Self(ptr)
    }

    pub fn get_pointer_byte_size(&self) -> u32 {
        use llvm_sys::target::LLVMPointerSize;

        unsafe { LLVMPointerSize(self.0) }
    }

    pub fn as_ref(&self) -> TargetDataRef {
        self.0
    }
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
    !is_any_library(options) && get_reloc_mode(options) == RelocModel::PIC
}

pub fn is_any_library(options: &Options) -> bool {
    options.project_type != ProjectType::Executable
}

pub fn get_reloc_mode(options: &Options) -> RelocModel {
    let arg = match options.codegen_opts.relocation_mode {
        Some(ref s) => &s[..],
        None => &options.target.options.relocation_model[..],
    };

    RelocModel::from_str(arg).unwrap()
}

pub fn get_tls_mode(options: &Options) -> ThreadLocalMode {
    let arg = match options.codegen_opts.tls_mode {
        Some(ref s) => &s[..],
        None => &options.target.options.tls_model[..],
    };

    ThreadLocalMode::from_str(arg).unwrap()
}

extern "C" {
    pub fn PrintTargetCPUs(target_machine: TargetMachineRef);
    pub fn PrintTargetFeatures(target_machine: TargetMachineRef);
    fn LLVMLumenCreateTargetMachine(config: &ffi::TargetMachineConfig) -> TargetMachineRef;
    fn LLVMLumenDisposeTargetMachine(T: TargetMachineRef);
    #[cfg(not(windows))]
    pub fn LLVMTargetMachineEmitToFileDescriptor(
        target_machine: TargetMachineRef,
        module: ModuleRef,
        fd: os::unix::io::RawFd,
        codegen: LLVMCodeGenFileType,
        error_message: *mut *mut libc::c_char,
    ) -> bool;

    #[cfg(windows)]
    pub fn LLVMTargetMachineEmitToFileDescriptor(
        target_machine: TargetMachineRef,
        module: ModuleRef,
        fd: os::windows::io::RawHandle,
        codegen: LLVMCodeGenFileType,
        error_message: *mut *mut libc::c_char,
    ) -> bool;
}
