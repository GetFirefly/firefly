use std::borrow::{Borrow, Cow};
use std::fmt;
use std::mem::MaybeUninit;
use std::ops::Deref;

use anyhow::anyhow;

use firefly_session::{Options, ProjectType};
use firefly_target::{CodeModel, Endianness, RelocModel};

use crate::codegen::{self, CodeGenFileType, CodeGenOptLevel};
use crate::ir::*;
use crate::support::{OwnedStringRef, StringRef};

extern "C" {
    type LlvmTargetMachine;
    type LlvmTarget;
    type LlvmTargetDataLayout;
}

/// Initialize all targets
pub fn init() {
    extern "C" {
        fn LLVM_InitializeAllTargetInfos();
        fn LLVM_InitializeAllTargets();
        fn LLVM_InitializeAllTargetMCs();
        fn LLVM_InitializeAllAsmPrinters();
        fn LLVM_InitializeAllAsmParsers();
        fn LLVM_InitializeAllDisassemblers();
    }
    unsafe {
        LLVM_InitializeAllTargetInfos();
        LLVM_InitializeAllTargets();
        LLVM_InitializeAllTargetMCs();
        LLVM_InitializeAllAsmPrinters();
        LLVM_InitializeAllAsmParsers();
        LLVM_InitializeAllDisassemblers();
    }
}

/// Returns the triple for the host machine as a string
pub fn default_triple() -> OwnedStringRef {
    extern "C" {
        fn LLVMGetDefaultTargetTriple() -> *const std::os::raw::c_char;
    }
    unsafe { OwnedStringRef::from_ptr(LLVMGetDefaultTargetTriple()) }
}

/// Normalizes the given triple, returning the normalized value as a new string
pub fn normalize_triple<S: Into<StringRef>>(triple: S) -> OwnedStringRef {
    extern "C" {
        fn LLVMNormalizeTargetTriple(
            triple: *const std::os::raw::c_char,
        ) -> *const std::os::raw::c_char;
    }
    let triple = triple.into();
    let c_str = triple.to_cstr();
    unsafe { OwnedStringRef::from_ptr(LLVMNormalizeTargetTriple(c_str.as_ptr())) }
}

/// Returns the host CPU as a string
pub fn host_cpu() -> OwnedStringRef {
    extern "C" {
        fn LLVMGetHostCPUName() -> *const std::os::raw::c_char;
    }
    unsafe { OwnedStringRef::from_ptr(LLVMGetHostCPUName()) }
}

/// Returns the specific cpu architecture being targeted as defined by the given compiler options
pub fn target_cpu(options: &Options) -> Cow<'static, str> {
    let name = match options.codegen_opts.target_cpu.as_ref().map(|s| s.as_str()) {
        Some(s) => s.to_string().into(),
        None => options.target.options.cpu.clone(),
    };

    if name != "native" {
        return name;
    }

    let native = host_cpu();
    native.to_string().into()
}

/// Returns the host's CPU features as a string
pub fn host_cpu_features() -> OwnedStringRef {
    extern "C" {
        fn LLVMGetHostCPUFeatures() -> *const std::os::raw::c_char;
    }
    unsafe { OwnedStringRef::from_ptr(LLVMGetHostCPUFeatures()) }
}

/// Iterates all of the default + custom target features defined by the provided compiler options
///
/// See firefly_target for the default features defined for each supported target
///
/// In addition to those defaults, manually-specified target features can be enabled via compiler flags
pub fn llvm_target_features(options: &Options) -> impl Iterator<Item = &str> {
    const FIREFLY_SPECIFIC_FEATURES: &[&str] = &["crt-static"];

    let cmdline = options
        .codegen_opts
        .target_features
        .as_ref()
        .map(|s| s.as_str())
        .unwrap_or("")
        .split(',')
        .filter(|f| !FIREFLY_SPECIFIC_FEATURES.iter().any(|s| f.contains(s)));
    options
        .target
        .options
        .features
        .split(',')
        .chain(cmdline)
        .filter(|l| l.is_empty())
}

/// Represents a reference to a specific codegen target
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Target(*const LlvmTarget);
impl Target {
    #[inline(always)]
    fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Get a target by name, e.g. x86-64 or wasm32
    pub fn by_name<S: Into<StringRef>>(name: S) -> Option<Target> {
        extern "C" {
            fn LLVMGetTargetFromName(name: *const std::os::raw::c_char) -> Target;
        }
        let name = name.into();
        let c_str = name.to_cstr();
        let target = unsafe { LLVMGetTargetFromName(c_str.as_ptr()) };
        if target.is_null() {
            None
        } else {
            Some(target)
        }
    }

    /// Get a target by its triple, e.g. aarch64-apple-darwin, wasm32-unknown-unknown
    pub fn by_triple<S: Into<StringRef>>(triple: S) -> Option<Target> {
        extern "C" {
            fn LLVMGetTargetFromTriple(
                triple: *const std::os::raw::c_char,
                t: *mut Target,
                error: *mut *mut std::os::raw::c_char,
            ) -> bool;
        }
        let triple = triple.into();
        let c_str = triple.to_cstr();
        let mut target = MaybeUninit::uninit();
        let failed = unsafe {
            LLVMGetTargetFromTriple(c_str.as_ptr(), target.as_mut_ptr(), core::ptr::null_mut())
        };
        if failed {
            None
        } else {
            Some(unsafe { target.assume_init() })
        }
    }

    /// Returns the name of this target
    pub fn name(self) -> StringRef {
        extern "C" {
            fn LLVMGetTargetName(t: Target) -> *const std::os::raw::c_char;
        }
        unsafe { StringRef::from_ptr(LLVMGetTargetName(self)) }
    }

    /// Returns the description of this target
    pub fn description(self) -> StringRef {
        extern "C" {
            fn LLVMGetTargetDescription(t: Target) -> *const std::os::raw::c_char;
        }
        unsafe { StringRef::from_ptr(LLVMGetTargetDescription(self)) }
    }

    /// Returns true if this target has a JIT
    pub fn has_jit(self) -> bool {
        extern "C" {
            fn LLVMTargetHasJIT(t: Target) -> bool;
        }
        unsafe { LLVMTargetHasJIT(self) }
    }

    /// Returns true if this target has a TargetMachine associated
    pub fn has_target_machine(self) -> bool {
        extern "C" {
            fn LLVMTargetHasTargetMachine(t: Target) -> bool;
        }
        unsafe { LLVMTargetHasTargetMachine(self) }
    }

    /// Returns true if this target has an ASM backend (required for emitting output)
    pub fn has_asm_backend(self) -> bool {
        extern "C" {
            fn LLVMTargetHasAsmBackend(t: Target) -> bool;
        }
        unsafe { LLVMTargetHasAsmBackend(self) }
    }

    pub fn iter() -> impl Iterator<Item = Target> {
        TargetIter::new()
    }
}

/// Used to walk the set of available targets
///
/// NOTE: Targets are more abstract than a TargetMachine, so there is not a direct
/// correlation from target to triple, as a target is often shared across many triples.
/// As a result, you can't walk the targets to get a list of triples. However, you can
/// walk the targets, and for each target print the available cpus/features for that target,
/// which can be used to select a specific target configuration (i.e. triple).
struct TargetIter(Target);
impl TargetIter {
    fn new() -> Self {
        extern "C" {
            fn LLVMGetFirstTarget() -> Target;
        }
        Self(unsafe { LLVMGetFirstTarget() })
    }
}
impl Iterator for TargetIter {
    type Item = Target;

    fn next(&mut self) -> Option<Self::Item> {
        extern "C" {
            fn LLVMGetNextTarget(t: Target) -> Target;
        }

        if self.0.is_null() {
            return None;
        }
        let next = unsafe { LLVMGetNextTarget(self.0) };
        if next.is_null() {
            self.0 = next;
            None
        } else {
            self.0 = next;
            Some(next)
        }
    }
}
impl std::iter::FusedIterator for TargetIter {}

/// Used to pass a target machine configuration to C++
#[repr(C)]
struct TargetMachineConfig {
    pub triple: StringRef,
    pub cpu: StringRef,
    pub abi: StringRef,
    pub features: *const StringRef,
    pub features_len: u32,
    pub relax_elf_relocations: bool,
    pub position_independent_code: bool,
    pub data_sections: bool,
    pub function_sections: bool,
    pub emit_stack_size_section: bool,
    pub preserve_asm_comments: bool,
    pub enable_threading: bool,
    pub code_model: *const CodeModel,
    pub reloc_model: *const RelocModel,
    pub opt_level: CodeGenOptLevel,
}

/// Represents a non-owning reference to an LLVM target machine
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct TargetMachine(*const LlvmTargetMachine);
impl TargetMachine {
    /// Creates a target machine from the current compiler options
    ///
    /// NOTE: This assumes the compiler options represent a valid target configuration, i.e.
    /// a target must have been selected, and the various configuration options must not be
    /// contradictory or invalid for the selected target. This is largely taken care of by the
    /// firefly_target crate, but it is worth calling out here. If the configuration is invalid,
    /// creation should fail and we'll raise an error, but if there are issues in LLVM itself it
    /// may cause issues with produced binaries.
    pub fn create(options: &Options) -> anyhow::Result<OwnedTargetMachine> {
        extern "C" {
            fn LLVMFireflyCreateTargetMachine(
                config: *const TargetMachineConfig,
                error: *mut *mut std::os::raw::c_char,
            ) -> TargetMachine;
        }
        crate::require_inited();

        let features_owned = llvm_target_features(options)
            .map(|f| f.to_string())
            .collect::<Vec<_>>();

        let triple: &str = options.target.llvm_target.borrow();
        let cpu = target_cpu(options);
        let cpu: &str = cpu.borrow();
        let abi: &str = options.target.options.llvm_abiname.borrow();
        let relax_elf_relocations = options.target.options.relax_elf_relocations;
        let default_reloc_model = options.target.options.relocation_model;
        let reloc_model = options
            .codegen_opts
            .relocation_model
            .unwrap_or(default_reloc_model);
        let position_independent_code =
            options.project_type == ProjectType::Executable && reloc_model == RelocModel::Pic;
        let function_sections = options.target.options.function_sections;
        let data_sections = function_sections;
        let emit_stack_size_section = options.debugging_opts.emit_stack_sizes;
        let preserve_asm_comments = options.debugging_opts.asm_comments;
        let enable_threading = if options.target.options.singlethread {
            // On the wasm target once the `atomics` feature is enabled that means that
            // we're no longer single-threaded, or otherwise we don't want LLVM to
            // lower atomic operations to single-threaded operations.
            if options.target.llvm_target.contains("wasm32")
                && features_owned.iter().any(|s| s == "+atomics")
            {
                true
            } else {
                false
            }
        } else {
            true
        };
        let default_code_model = options.target.options.code_model;
        let code_model = options.codegen_opts.code_model.or(default_code_model);
        let code_model_ptr = code_model
            .as_ref()
            .map(|cm| cm as *const CodeModel)
            .unwrap_or(std::ptr::null());
        let (opt_level, _) = codegen::to_llvm_opt_settings(options.opt_level);

        let features = features_owned
            .iter()
            .map(StringRef::from)
            .collect::<Vec<_>>();

        let config = TargetMachineConfig {
            triple: StringRef::from(triple),
            cpu: StringRef::from(cpu),
            abi: StringRef::from(abi),
            features: features.as_ptr(),
            features_len: features.len().try_into().unwrap(),
            relax_elf_relocations,
            position_independent_code,
            data_sections,
            function_sections,
            emit_stack_size_section,
            preserve_asm_comments,
            enable_threading,
            code_model: code_model_ptr,
            reloc_model: &reloc_model,
            opt_level,
        };

        let mut error = MaybeUninit::uninit();
        let tm = unsafe { LLVMFireflyCreateTargetMachine(&config, error.as_mut_ptr()) };
        if tm.is_null() {
            let error = unsafe { OwnedStringRef::from_ptr(error.assume_init()) };
            Err(anyhow!(
                "Target machine creation for {} failed: {}",
                triple,
                &error
            ))
        } else {
            Ok(OwnedTargetMachine(tm))
        }
    }

    #[inline]
    fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Returns the target for this target machine
    pub fn target(self) -> Target {
        extern "C" {
            fn LLVMGetTargetMachineTarget(t: TargetMachine) -> Target;
        }
        unsafe { LLVMGetTargetMachineTarget(self) }
    }

    /// Returns the triple used in creating this target machine
    pub fn triple(self) -> StringRef {
        extern "C" {
            fn LLVMGetTargetMachineTriple(t: TargetMachine) -> *const std::os::raw::c_char;
        }
        unsafe { StringRef::from_ptr(LLVMGetTargetMachineTriple(self)) }
    }

    /// Returns the cpu used in creating this target machine
    pub fn cpu(self) -> StringRef {
        extern "C" {
            fn LLVMGetTargetMachineCPU(t: TargetMachine) -> *const std::os::raw::c_char;
        }
        unsafe { StringRef::from_ptr(LLVMGetTargetMachineCPU(self)) }
    }

    /// Returns the feature string used in creating this target machine
    pub fn features(self) -> StringRef {
        extern "C" {
            fn LLVMGetTargetMachineFeatureString(t: TargetMachine) -> *const std::os::raw::c_char;
        }
        unsafe { StringRef::from_ptr(LLVMGetTargetMachineFeatureString(self)) }
    }

    /// Prints the available set of target CPUs to stdout
    pub fn print_target_cpus(self) {
        extern "C" {
            fn PrintTargetCPUs(tm: TargetMachine);
        }

        unsafe { PrintTargetCPUs(self) }
    }

    /// Prints the available set of target features to stdout
    pub fn print_target_features(self) {
        extern "C" {
            fn PrintTargetFeatures(tm: TargetMachine);
        }

        unsafe { PrintTargetFeatures(self) };
    }

    /// Get a reference to the target data layout for this target
    pub fn data_layout(self) -> TargetDataLayout {
        extern "C" {
            fn LLVMCreateTargetDataLayout(t: TargetMachine) -> TargetDataLayout;
        }
        let ptr = unsafe { LLVMCreateTargetDataLayout(self) };
        assert!(!ptr.is_null());
        ptr
    }

    /// Sets the target machine's ASM verbosity
    pub fn set_asm_verbosity(self, verbose: bool) {
        extern "C" {
            fn LLVMSetTargetMachineAsmVerbosity(t: TargetMachine, verbose: bool);
        }
        unsafe { LLVMSetTargetMachineAsmVerbosity(self, verbose) }
    }

    pub fn emit_to_file<S: Into<StringRef>>(
        self,
        module: Module,
        filename: S,
        codegen: CodeGenFileType,
    ) -> anyhow::Result<()> {
        extern "C" {
            fn LLVMTargetMachineEmitToFile(
                t: TargetMachine,
                m: Module,
                filename: *const i8,
                codegen: CodeGenFileType,
                error: *mut *mut std::os::raw::c_char,
            ) -> bool;
        }

        let filename = filename.into();
        let filename = filename.to_cstr();
        let mut error = MaybeUninit::uninit();
        let failed = unsafe {
            LLVMTargetMachineEmitToFile(
                self,
                module,
                filename.as_ptr(),
                codegen,
                error.as_mut_ptr(),
            )
        };
        if failed {
            let error = unsafe { OwnedStringRef::from_ptr(error.assume_init()) };
            Err(anyhow!("{}", &error))
        } else {
            Ok(())
        }
    }

    #[cfg(not(windows))]
    pub fn emit_to_fd(
        self,
        module: Module,
        fd: std::os::unix::io::RawFd,
        codegen: CodeGenFileType,
    ) -> anyhow::Result<()> {
        extern "C" {
            fn LLVMTargetMachineEmitToFileDescriptor(
                t: TargetMachine,
                module: Module,
                fd: std::os::unix::io::RawFd,
                codegen: CodeGenFileType,
                error: *mut *mut std::os::raw::c_char,
            ) -> bool;
        }

        let mut error = MaybeUninit::uninit();
        let failed = unsafe {
            LLVMTargetMachineEmitToFileDescriptor(self, module, fd, codegen, error.as_mut_ptr())
        };
        if failed {
            let error = unsafe { OwnedStringRef::from_ptr(error.assume_init()) };
            Err(anyhow!("{}", &error))
        } else {
            Ok(())
        }
    }

    #[cfg(windows)]
    pub fn emit_to_fd(
        self,
        module: Module,
        fd: std::os::windows::io::RawHandle,
        codegen: CodeGenFileType,
    ) -> anyhow::Result<()> {
        extern "C" {
            fn LLVMTargetMachineEmitToFileDescriptor(
                t: TargetMachine,
                module: Module,
                fd: std::os::windows::io::RawHandle,
                codegen: CodeGenFileType,
                error: *mut *mut std::os::raw::c_char,
            ) -> bool;
        }

        let mut error = MaybeUninit::uninit();
        let failed = unsafe {
            LLVMTargetMachineEmitToFileDescriptor(self, module, fd, codegen, error.as_mut_ptr())
        };
        if failed {
            let error = unsafe { OwnedStringRef::from_ptr(error.assume_init()) };
            Err(anyhow!("{}", &error))
        } else {
            Ok(())
        }
    }
}
impl Eq for TargetMachine {}
impl PartialEq for TargetMachine {
    fn eq(&self, other: &Self) -> bool {
        core::ptr::eq(self.0, other.0)
    }
}
impl fmt::Pointer for TargetMachine {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TargetMachine({:p})", self.0)
    }
}
impl fmt::Debug for TargetMachine {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TargetMachine({:p})", self.0)
    }
}

/// Represents an owned reference to a TargetMachine
#[repr(transparent)]
pub struct OwnedTargetMachine(TargetMachine);
impl OwnedTargetMachine {
    /// Returns the underlying TargetMachine handle
    ///
    /// NOTE: It is up to the caller to ensure that the handle does not outlive this struct
    pub fn handle(&self) -> TargetMachine {
        self.0
    }
}
unsafe impl Send for TargetMachine {}
unsafe impl Sync for TargetMachine {}
impl Eq for OwnedTargetMachine {}
impl PartialEq for OwnedTargetMachine {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl fmt::Pointer for OwnedTargetMachine {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", self.0)
    }
}
impl fmt::Debug for OwnedTargetMachine {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}
impl Deref for OwnedTargetMachine {
    type Target = TargetMachine;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Drop for OwnedTargetMachine {
    fn drop(&mut self) {
        extern "C" {
            fn LLVMDisposeTargetMachine(t: TargetMachine);
        }
        unsafe {
            LLVMDisposeTargetMachine(self.0);
        }
    }
}

/// Represents the target data layout for some target
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct TargetDataLayout(*const LlvmTargetDataLayout);
impl TargetDataLayout {
    #[inline(always)]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Creates target data from the given target layout string
    pub fn new<S: Into<StringRef>>(layout: S) -> Result<OwnedTargetDataLayout, ()> {
        extern "C" {
            fn LLVMCreateTargetData(layout: *const std::os::raw::c_char) -> TargetDataLayout;
        }
        let layout = layout.into();
        let c_str = layout.to_cstr();
        let ptr = unsafe { LLVMCreateTargetData(c_str.as_ptr()) };
        if ptr.is_null() {
            Err(())
        } else {
            Ok(OwnedTargetDataLayout(ptr))
        }
    }

    /// Returns the endianness of this target's data layout
    pub fn byte_order(self) -> Endianness {
        extern "C" {
            fn LLVMByteOrder(t: TargetDataLayout) -> u8;
        }
        match unsafe { LLVMByteOrder(self) } {
            0 => Endianness::Big,
            1 => Endianness::Little,
            n => panic!("invalid byte ordering variant {}", n),
        }
    }

    /// Gets the size of pointers in bytes, e.g. 8 for x86_64
    pub fn get_pointer_byte_size(self) -> usize {
        extern "C" {
            fn LLVMPointerSize(t: TargetDataLayout) -> u32;
        }

        unsafe { LLVMPointerSize(self) as usize }
    }

    /// Gets the LLVM type representing a pointer-sized integer for this data layout
    pub fn get_int_ptr_type(self, context: Context) -> IntegerType {
        extern "C" {
            fn LLVMIntPtrTypeInContext(context: Context, t: TargetDataLayout) -> IntegerType;
        }
        unsafe { LLVMIntPtrTypeInContext(context, self) }
    }

    /// Provides sizeof facilities for the given LLVM type in this data layout
    pub fn size_of_type<T: Type>(self, ty: T) -> usize {
        extern "C" {
            fn LLVMSizeOfTypeInBits(t: TargetDataLayout, ty: TypeBase) -> u64;
        }
        unsafe { LLVMSizeOfTypeInBits(self, ty.base()) }
            .try_into()
            .unwrap()
    }

    /// Provides the ABI-defined alignment of the given type in this data layout
    pub fn abi_alignment_of_type<T: Type>(self, ty: T) -> usize {
        extern "C" {
            fn LLVMABIAlignmentOfType(t: TargetDataLayout, ty: TypeBase) -> u32;
        }
        unsafe { LLVMABIAlignmentOfType(self, ty.base()) as usize }
    }
}
impl fmt::Display for TargetDataLayout {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        extern "C" {
            fn LLVMCopyStringRepOfTargetData(t: TargetDataLayout) -> *const std::os::raw::c_char;
        }
        let rep = unsafe { OwnedStringRef::from_ptr(LLVMCopyStringRepOfTargetData(*self)) };
        write!(f, "{}", &rep)
    }
}

/// Represents an owned reference to an llvm::TargetData instance
pub struct OwnedTargetDataLayout(TargetDataLayout);
impl Deref for OwnedTargetDataLayout {
    type Target = TargetDataLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Drop for OwnedTargetDataLayout {
    fn drop(&mut self) {
        extern "C" {
            fn LLVMDisposeTargetData(t: TargetDataLayout);
        }
        unsafe { LLVMDisposeTargetData(self.0) }
    }
}
impl fmt::Display for OwnedTargetDataLayout {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.0)
    }
}
