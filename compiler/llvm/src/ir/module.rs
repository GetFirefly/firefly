use std::ffi::CStr;
use std::fmt;
use std::mem::MaybeUninit;
use std::ops::Deref;

use anyhow::anyhow;
use liblumen_util::{self as util, emit::Emit};

use crate::codegen::CodeGenFileType;
use crate::support::*;
use crate::target::{TargetDataLayout, TargetMachine};

use super::*;

extern "C" {
    type LlvmModule;
    type LlvmModuleFlag;
}

/// Represents a borrowed reference to an LLVM module
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Module(*const LlvmModule);
impl Module {
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    pub fn context(self) -> Context {
        extern "C" {
            fn LLVMGetModuleContext(m: Module) -> Context;
        }
        unsafe { LLVMGetModuleContext(self) }
    }

    pub fn name(self) -> StringRef {
        extern "C" {
            fn LLVMGetModuleIdentifier(m: Module, len: *mut usize) -> *const u8;
        }
        let mut len = MaybeUninit::uninit();
        let bytes = unsafe {
            let ptr = LLVMGetModuleIdentifier(self, len.as_mut_ptr());
            core::slice::from_raw_parts(ptr, len.assume_init())
        };
        bytes.into()
    }

    pub fn set_name<S: Into<StringRef>>(self, name: S) {
        extern "C" {
            fn LLVMSetModuleIdentifier(m: Module, name: *const u8, len: usize);
        }
        let name = name.into();
        unsafe { LLVMSetModuleIdentifier(self, name.data, name.len) }
    }

    pub fn source_file(self) -> StringRef {
        extern "C" {
            fn LLVMGetSourceFileName(m: Module, len: *mut usize) -> *const u8;
        }
        let mut len = MaybeUninit::uninit();
        let bytes = unsafe {
            let ptr = LLVMGetSourceFileName(self, len.as_mut_ptr());
            core::slice::from_raw_parts(ptr, len.assume_init())
        };
        bytes.into()
    }

    pub fn set_source_file<S: Into<StringRef>>(self, filename: S) {
        extern "C" {
            fn LLVMSetSourceFileName(m: Module, name: *const u8, len: usize);
        }
        let name = filename.into();
        unsafe { LLVMSetSourceFileName(self, name.data, name.len) }
    }

    pub fn data_layout_str(self) -> Option<StringRef> {
        extern "C" {
            fn LLVMGetDataLayoutStr(m: Module) -> *const std::os::raw::c_char;
        }
        let ptr = unsafe { LLVMGetDataLayoutStr(self) };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { CStr::from_ptr(ptr) }.into())
        }
    }

    pub fn set_data_layout_str<S: Into<StringRef>>(self, layout: S) {
        extern "C" {
            fn LLVMSetDataLayout(m: Module, layout: *const std::os::raw::c_char);
        }
        let layout = layout.into();
        let c_str = layout.to_cstr();
        unsafe { LLVMSetDataLayout(self, c_str.as_ptr()) }
    }

    pub fn data_layout(self) -> Option<TargetDataLayout> {
        extern "C" {
            fn LLVMGetModuleDataLayout(m: Module) -> TargetDataLayout;
        }
        let layout = unsafe { LLVMGetModuleDataLayout(self) };
        if layout.is_null() {
            None
        } else {
            Some(layout)
        }
    }

    pub fn set_data_layout(self, layout: TargetDataLayout) {
        extern "C" {
            fn LLVMSetModuleDataLayout(m: Module, layout: TargetDataLayout);
        }
        unsafe { LLVMSetModuleDataLayout(self, layout) }
    }

    pub fn target_triple(self) -> Option<StringRef> {
        extern "C" {
            fn LLVMGetTarget(m: Module) -> *const std::os::raw::c_char;
        }
        let ptr = unsafe { LLVMGetTarget(self) };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { CStr::from_ptr(ptr) }.into())
        }
    }

    pub fn set_target_triple<S: Into<StringRef>>(self, triple: S) {
        extern "C" {
            fn LLVMSetTarget(m: Module, triple: *const std::os::raw::c_char);
        }
        let triple = triple.into();
        let c_str = triple.to_cstr();
        unsafe { LLVMSetTarget(self, c_str.as_ptr()) }
    }

    pub fn flags(self) -> ModuleFlags {
        extern "C" {
            fn LLVMCopyModuleFlagsMetadata(m: Module, len: *mut usize) -> *const LlvmModuleFlag;
        }
        let mut len = MaybeUninit::uninit();
        unsafe {
            let ptr = LLVMCopyModuleFlagsMetadata(self, len.as_mut_ptr());
            ModuleFlags::new(ptr, len.assume_init())
        }
    }

    /// Returns an iterator over this module's named metadata names
    pub fn named_metadata(self) -> impl Iterator<Item = StringRef> {
        NamedMetadataIter::new(self)
    }

    /// Gets the first named metadata node with the given name in this module
    pub fn get_named_metadata<S: Into<StringRef>>(self, name: S) -> Vec<MetadataValue> {
        extern "C" {
            fn LLVMGetNamedMetadataNumOperands(m: Module, name: *const std::os::raw::c_char)
                -> u32;
            fn LLVMGetNamedMetadataOperands(
                m: Module,
                name: *const std::os::raw::c_char,
                operands: *mut MetadataValue,
            );
        }
        let name = name.into();
        let c_str = name.to_cstr();
        let len = unsafe { LLVMGetNamedMetadataNumOperands(self, c_str.as_ptr()) } as usize;
        let mut operands = Vec::with_capacity(len);
        unsafe {
            LLVMGetNamedMetadataOperands(self, c_str.as_ptr(), operands.as_mut_ptr());
            operands.set_len(len);
        }
        operands
    }

    /// Gets the flag with the given key in this module, if it exists
    pub fn get_module_flag<S: Into<StringRef>>(self, key: S) -> Option<Metadata> {
        extern "C" {
            fn LLVMGetModuleFlag(m: Module, key: *const u8, len: usize) -> Metadata;
        }
        let key = key.into();
        let meta = unsafe { LLVMGetModuleFlag(self, key.data, key.len) };
        if meta.is_null() {
            None
        } else {
            Some(meta)
        }
    }

    /// Adds a flag with the given key to this module
    #[inline]
    pub fn set_module_flag<S: Into<StringRef>>(
        self,
        key: S,
        value: Metadata,
        behavior: ModuleFlagBehavior,
    ) {
        extern "C" {
            fn LLVMAddModuleFlag(
                m: Module,
                behavior: ModuleFlagBehavior,
                key: *const u8,
                key_len: usize,
                value: Metadata,
            );
        }
        let key = key.into();
        unsafe {
            LLVMAddModuleFlag(self, behavior, key.data, key.len, value);
        }
    }

    pub fn functions(&self) -> impl Iterator<Item = Function> {
        FunctionsIter::new(*self)
    }

    pub fn get_function<S: Into<StringRef>>(self, name: S) -> Option<Function> {
        extern "C" {
            fn LLVMGetNamedFunction(m: Module, name: *const std::os::raw::c_char) -> Function;
        }
        let name = name.into();
        let c_str = name.to_cstr();
        let value = unsafe { LLVMGetNamedFunction(self, c_str.as_ptr()) };
        if value.is_null() {
            None
        } else {
            Some(value)
        }
    }

    pub fn add_function<S: Into<StringRef>>(self, name: S, ty: FunctionType) -> Function {
        extern "C" {
            fn LLVMAddFunction(
                m: Module,
                name: *const std::os::raw::c_char,
                ty: FunctionType,
            ) -> Function;
        }

        let name = name.into();
        let c_str = name.to_cstr();
        let value = unsafe { LLVMAddFunction(self, c_str.as_ptr(), ty) };
        assert!(!value.is_null());
        value
    }

    pub fn get_or_add_function<S: Into<StringRef>>(self, name: S, ty: FunctionType) -> Function {
        let name = name.into();
        self.get_function(name)
            .unwrap_or_else(|| self.add_function(name, ty))
    }

    pub fn get_global<S: Into<StringRef>>(self, name: S) -> Option<GlobalVariable> {
        extern "C" {
            fn LLVMGetNamedGlobal(
                module: Module,
                name: *const std::os::raw::c_char,
            ) -> GlobalVariable;
        }
        let name = name.into();
        let c_str = name.to_cstr();
        let global = unsafe { LLVMGetNamedGlobal(self, c_str.as_ptr()) };
        if global.is_null() {
            None
        } else {
            Some(global)
        }
    }

    pub fn get_or_add_global<S: Into<StringRef>, T: Type>(
        self,
        ty: T,
        name: S,
        initializer: Option<ValueBase>,
    ) -> GlobalVariable {
        let name = name.into();
        if let Some(gv) = self.get_global(name) {
            gv
        } else {
            self.add_global(ty, name, initializer)
        }
    }

    /// Adds a new global with the given name, type, and initializer to this module
    pub fn add_global<S: Into<StringRef>, T: Type>(
        self,
        ty: T,
        name: S,
        initializer: Option<ValueBase>,
    ) -> GlobalVariable {
        extern "C" {
            fn LLVMAddGlobal(
                module: Module,
                ty: TypeBase,
                name: *const std::os::raw::c_char,
            ) -> GlobalVariable;
        }

        let name = name.into();
        let c_str = name.to_cstr();
        let global = unsafe { LLVMAddGlobal(self, ty.base(), c_str.as_ptr()) };
        if let Some(init) = initializer {
            global.set_initializer(init);
        }
        global
    }

    /// Gets the COMDAT in this module with the given name.
    /// It is created if it doesn't yet exist.
    pub fn get_or_add_comdat<S: Into<StringRef>>(self, name: S) -> Comdat {
        extern "C" {
            fn LLVMGetOrInsertComdat(module: Module, name: *const std::os::raw::c_char) -> Comdat;
        }
        let name = name.into();
        let c_str = name.to_cstr();
        unsafe { LLVMGetOrInsertComdat(self, c_str.as_ptr()) }
    }

    pub fn verify(self) -> anyhow::Result<()> {
        // 0 = print to stderr + abort, 1 = print to stderr + return true, 2 = no output, return true
        const ON_VERIFY_FAILED_RETURN_STATUS: u32 = 2;

        extern "C" {
            fn LLVMVerifyModule(
                m: Module,
                action: u32,
                error: *mut *mut std::os::raw::c_char,
            ) -> bool;
        }

        let mut error = MaybeUninit::uninit();
        let failed =
            unsafe { LLVMVerifyModule(self, ON_VERIFY_FAILED_RETURN_STATUS, error.as_mut_ptr()) };

        if failed {
            let module_name = self.name();
            let error = unsafe { OwnedStringRef::from_ptr(error.assume_init()) };
            Err(anyhow!(format!(
                "failed to verify {}: {}",
                module_name, &error
            )))
        } else {
            Ok(())
        }
    }

    /// Strip debug info in this module, if it exists.
    ///
    /// Returns true if the module is modified.
    pub fn strip_debug_info(self) -> bool {
        extern "C" {
            fn LLVMStripModuleDebugInfo(module: Module) -> bool;
        }
        unsafe { LLVMStripModuleDebugInfo(self) }
    }

    /// Dump a debug representation of this module to stderr
    pub fn dump(self) {
        extern "C" {
            fn LLVMDumpModule(m: Module);
        }
        unsafe { LLVMDumpModule(self) }
    }

    /// Write this module as LLVM IR to the given file path
    pub fn print_to_file<S: Into<StringRef>>(self, filename: S) -> anyhow::Result<()> {
        extern "C" {
            fn LLVMPrintModuleToFile(
                m: Module,
                filename: *const std::os::raw::c_char,
                error: *mut *mut std::os::raw::c_char,
            ) -> bool;
        }
        let filename = filename.into();
        let c_str = filename.to_cstr();
        let mut error = MaybeUninit::uninit();
        let failed = unsafe { LLVMPrintModuleToFile(self, c_str.as_ptr(), error.as_mut_ptr()) };
        if failed {
            let error = unsafe { OwnedStringRef::from_ptr(error.assume_init()) };
            Err(anyhow!("{}", &error))
        } else {
            Ok(())
        }
    }

    /// Write this module as LLVM IR to the given file
    pub fn emit_ir(self, f: &mut std::fs::File) -> anyhow::Result<()> {
        let fd = util::fs::get_file_descriptor(f);
        let mut error = MaybeUninit::uninit();
        let failed = unsafe { LLVMEmitToFileDescriptor(self, fd, error.as_mut_ptr()) };

        if failed {
            let error = unsafe { OwnedStringRef::from_ptr(error.assume_init()) };
            Err(anyhow!("{}", &error))
        } else {
            Ok(())
        }
    }

    /// Write this module as LLVM bitcode (i.e. 'foo.bc') to the given file
    pub fn emit_bc(self, f: &mut std::fs::File) -> anyhow::Result<()> {
        let fd = util::fs::get_file_descriptor(f);
        let mut error = MaybeUninit::uninit();
        let failed = unsafe { LLVMEmitBitcodeToFileDescriptor(self, fd, error.as_mut_ptr()) };

        if failed {
            let error = unsafe { OwnedStringRef::from_ptr(error.assume_init()) };
            Err(anyhow!("{}", &error))
        } else {
            Ok(())
        }
    }

    /// Generate textual target-specific assembly from this module using the given TargetMachine, writing it to the given file
    ///
    /// The assembly generated by this function is generally written to files with a `.s` extension.
    pub fn emit_asm(
        self,
        f: &mut std::fs::File,
        target_machine: TargetMachine,
    ) -> anyhow::Result<()> {
        self.emit_file(f, target_machine, CodeGenFileType::Assembly)
    }

    /// Generate object code for this module to the given file, ready for consumption by a linker.
    ///
    /// The code generated by this function is generally written to files with a `.o` extension
    pub fn emit_obj(
        self,
        f: &mut std::fs::File,
        target_machine: TargetMachine,
    ) -> anyhow::Result<()> {
        self.emit_file(f, target_machine, CodeGenFileType::Object)
    }

    fn emit_file(
        self,
        f: &mut std::fs::File,
        target_machine: TargetMachine,
        file_type: CodeGenFileType,
    ) -> anyhow::Result<()> {
        target_machine.emit_to_fd(self, util::fs::get_file_descriptor(f), file_type)
    }
}
impl Eq for Module {}
impl PartialEq for Module {
    fn eq(&self, other: &Self) -> bool {
        core::ptr::eq(self.0, other.0)
    }
}
impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        extern "C" {
            fn LLVMPrintModuleToString(m: Module) -> *const std::os::raw::c_char;
        }
        let ptr = unsafe { OwnedStringRef::from_ptr(LLVMPrintModuleToString(*self)) };
        write!(f, "{}", &ptr)
    }
}
// The default implementation of `emit` is LLVM IR
impl Emit for Module {
    fn file_type(&self) -> Option<&'static str> {
        Some("ll")
    }

    fn emit(&self, f: &mut std::fs::File) -> anyhow::Result<()> {
        self.emit_ir(f)
    }
}

/// Represents an owned reference to an LLVM module
#[repr(transparent)]
pub struct OwnedModule(Module);
unsafe impl Send for OwnedModule {}
unsafe impl Sync for OwnedModule {}
impl OwnedModule {
    #[inline]
    pub fn as_ref(&self) -> Module {
        self.0
    }

    #[inline]
    pub fn as_mut(&mut self) -> Module {
        self.0
    }
}
impl Clone for OwnedModule {
    fn clone(&self) -> Self {
        extern "C" {
            fn LLVMCloneModule(m: Module) -> OwnedModule;
        }
        unsafe { LLVMCloneModule(self.0) }
    }
}
impl Deref for OwnedModule {
    type Target = Module;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Drop for OwnedModule {
    fn drop(&mut self) {
        extern "C" {
            fn LLVMDisposeModule(m: Module);
        }
        unsafe { LLVMDisposeModule(self.0) }
    }
}
impl Eq for OwnedModule {}
impl PartialEq for OwnedModule {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl fmt::Display for OwnedModule {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.0)
    }
}
impl fmt::Debug for OwnedModule {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.0)
    }
}
// The default implementation of `emit` is LLVM IR
impl Emit for OwnedModule {
    fn file_type(&self) -> Option<&'static str> {
        Some("ll")
    }

    fn emit(&self, f: &mut std::fs::File) -> anyhow::Result<()> {
        self.emit_ir(f)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModuleFlagBehavior {
    /// Emits an error if two values disagree, otherwise the resulting value is that of the
    /// operands.
    Error,
    /// Emits a warning if two values disagree. The result value will be the operand for the flag
    /// from the first module being linked.
    Warning,
    /// Adds a requirement that another module flag be present and have a specified value after
    /// linking is performed. The value must be a metadata pair, where the first element of the
    /// pair is the ID of the module flag to be restricted, and the second element of the pair is
    /// the value the module flag should be restricted to. This behavior can be used to restrict
    /// the allowable results (via triggering of an error) of linking IDs with the **Override**
    /// behavior.
    Require,
    /// Uses the specified value, regardless of the behavior or value of the other module. If both
    /// modules specify **Override**, but the values differ, an error will be emitted.
    Override,
    /// Appends the two values, which are required to be metadata nodes.
    Append,
    /// Appends the two values, which are required to be metadata nodes. However, duplicate entries
    /// in the second list are dropped during the append operation.
    AppendUnique,
}

/// Represents an owned reference to a set of module flags
pub struct ModuleFlags {
    data: *const LlvmModuleFlag,
    len: usize,
}
impl ModuleFlags {
    unsafe fn new(data: *const LlvmModuleFlag, len: usize) -> Self {
        Self { data, len }
    }

    /// Gets an iterator over all flags in the set
    #[inline]
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = ModuleFlag<'a>> + 'a {
        ModuleFlagIter::new(self)
    }

    /// Gets the flag at the given index
    pub fn get(&self, index: usize) -> ModuleFlag<'_> {
        assert!(self.len > index);
        ModuleFlag {
            flags: self,
            index: index.try_into().unwrap(),
        }
    }
}
impl Drop for ModuleFlags {
    fn drop(&mut self) {
        extern "C" {
            fn LLVMDisposeModuleFlagsMetadata(ptr: *const LlvmModuleFlag);
        }
        unsafe { LLVMDisposeModuleFlagsMetadata(self.data) }
    }
}

/// Represents a specific flag associated with a module
pub struct ModuleFlag<'a> {
    flags: &'a ModuleFlags,
    index: u32,
}
impl<'a> ModuleFlag<'a> {
    /// Returns the flag behavior for this flag
    pub fn behavior(&self) -> ModuleFlagBehavior {
        extern "C" {
            fn LLVMModuleFlagEntriesGetFlagBehavior(
                ptr: *const LlvmModuleFlag,
                index: u32,
            ) -> ModuleFlagBehavior;
        }
        unsafe { LLVMModuleFlagEntriesGetFlagBehavior(self.flags.data, self.index) }
    }

    /// Return the key for this module flag
    pub fn key(&self) -> StringRef {
        extern "C" {
            fn LLVMModuleFlagEntriesGetKey(
                ptr: *const LlvmModuleFlag,
                index: u32,
                len: *mut usize,
            ) -> *const u8;
        }
        let mut len = MaybeUninit::uninit();
        let ptr =
            unsafe { LLVMModuleFlagEntriesGetKey(self.flags.data, self.index, len.as_mut_ptr()) };
        assert!(!ptr.is_null());
        let bytes = unsafe { core::slice::from_raw_parts(ptr, len.assume_init()) };
        bytes.into()
    }

    /// Returns the metadata value associated with this flag
    pub fn metadata(&self) -> Metadata {
        extern "C" {
            fn LLVMModuleFlagEntriesGetMetadata(ptr: *const LlvmModuleFlag, index: u32)
                -> Metadata;
        }
        unsafe { LLVMModuleFlagEntriesGetMetadata(self.flags.data, self.index) }
    }
}

struct ModuleFlagIter<'a> {
    flags: &'a ModuleFlags,
    pos: usize,
}
impl<'a> ModuleFlagIter<'a> {
    fn new(flags: &'a ModuleFlags) -> Self {
        Self { flags, pos: 0 }
    }
}
impl<'a> Iterator for ModuleFlagIter<'a> {
    type Item = ModuleFlag<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.flags.len {
            return None;
        }
        let flag = self.flags.get(self.pos);
        self.pos += 1;
        Some(flag)
    }
}

struct FunctionsIter(Function);
impl FunctionsIter {
    fn new(m: Module) -> Self {
        extern "C" {
            fn LLVMGetFirstFunction(m: Module) -> Function;
        }
        Self(unsafe { LLVMGetFirstFunction(m) })
    }
}
impl Iterator for FunctionsIter {
    type Item = Function;

    fn next(&mut self) -> Option<Self::Item> {
        extern "C" {
            fn LLVMGetNextFunction(v: Function) -> Function;
        }

        if self.0.is_null() {
            return None;
        }

        let next = self.0;
        self.0 = unsafe { LLVMGetNextFunction(next) };
        Some(next)
    }
}

extern "C" {
    #[cfg(not(windows))]
    pub fn LLVMEmitToFileDescriptor(
        m: Module,
        fd: std::os::unix::io::RawFd,
        error_message: *mut *mut std::os::raw::c_char,
    ) -> bool;

    #[cfg(windows)]
    pub fn LLVMEmitToFileDescriptor(
        m: Module,
        fd: std::os::windows::io::RawHandle,
        error_message: *mut *mut std::os::raw::c_char,
    ) -> bool;

    #[cfg(not(windows))]
    pub fn LLVMEmitBitcodeToFileDescriptor(
        m: Module,
        fd: std::os::unix::io::RawFd,
        error_message: *mut *mut std::os::raw::c_char,
    ) -> bool;

    #[cfg(windows)]
    pub fn LLVMEmitBitcodeToFileDescriptor(
        m: Module,
        fd: std::os::windows::io::RawHandle,
        error_message: *mut *mut std::os::raw::c_char,
    ) -> bool;
}
