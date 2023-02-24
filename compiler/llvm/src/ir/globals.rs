use std::mem::MaybeUninit;

use super::*;
use crate::support::StringRef;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Linkage {
    /// Externally visible
    External,
    AvailableExternally,
    /// Keep one copy of function when linking (inline)
    LinkOnceAny,
    /// Keep one copy of function when linking (inline), but only replaced by something equivalent
    LinkOnceODR,
    #[deprecated]
    LinkOnceODRAutoHide,
    /// Keep one copy of function when linking (weak)
    WeakAny,
    /// Keep one copy of function when linking (weak), but only replaced by something equivalent
    WeakODR,
    /// Special purpose, only applies to global arrays (e.g. global constructors)
    Appending,
    /// Rename collisions when linking (static functions)
    Internal,
    /// Like internal, but omit from symbol table
    Private,
    #[deprecated]
    DLLImport,
    #[deprecated]
    DLLExport,
    /// Externally visible (weak)
    ExternalWeak,
    #[deprecated]
    Ghost,
    /// Tentative definitions
    Common,
    /// Like private, but linker removes
    LinkerPrivate,
    /// Like linker-private, but weak
    LinkerPrivateWeak,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Visibility {
    Public = 0,
    Hidden,
    Protected,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum UnnamedAddr {
    /// The address of the global value is significant
    No = 0,
    /// The address of the global value is locally insignificant
    Local,
    /// The address of the global value is globally insignificant
    Global,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ThreadLocalMode {
    NotThreadLocal = 0,
    GeneralDynamic,
    LocalDynamic,
    InitialExec,
    LocalExec,
}
impl Default for ThreadLocalMode {
    fn default() -> Self {
        Self::NotThreadLocal
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DLLStorageClass {
    Default = 0,
    /// Function to be imported from DLL
    Import,
    /// Function to be accessible from DLL
    Export,
}

/// This trait corresponds to the llvm::Align class and its subtypes
pub trait Align: Value {
    fn alignment(&self) -> usize {
        extern "C" {
            fn LLVMGetAlignment(value: ValueBase) -> u32;
        }
        unsafe { LLVMGetAlignment(self.base()) as usize }
    }
    fn set_alignment(&self, alignment: usize) {
        extern "C" {
            fn LLVMSetAlignment(value: ValueBase, alignment: u32);
        }
        unsafe { LLVMSetAlignment(self.base(), alignment.try_into().unwrap()) }
    }
}
impl<T: GlobalValue> Align for T {}

/// This trait corresponds to the llvm::GlobalValue class and its subtypes
pub trait GlobalValue: Value {
    fn parent(&self) -> Module {
        extern "C" {
            fn LLVMGetGlobalParent(value: ValueBase) -> Module;
        }
        unsafe { LLVMGetGlobalParent(self.base()) }
    }

    fn is_declaration(&self) -> bool {
        extern "C" {
            fn LLVMIsDeclaration(value: ValueBase) -> bool;
        }
        unsafe { LLVMIsDeclaration(self.base()) }
    }

    fn linkage(&self) -> Linkage {
        extern "C" {
            fn LLVMGetLinkage(value: ValueBase) -> Linkage;
        }
        unsafe { LLVMGetLinkage(self.base()) }
    }

    fn set_linkage(&self, linkage: Linkage) {
        extern "C" {
            fn LLVMSetLinkage(value: ValueBase, linkage: Linkage);
        }
        unsafe { LLVMSetLinkage(self.base(), linkage) }
    }

    fn section(&self) -> StringRef {
        extern "C" {
            fn LLVMGetSection(value: ValueBase) -> *const std::os::raw::c_char;
        }
        unsafe { StringRef::from_ptr(LLVMGetSection(self.base())) }
    }

    fn set_section<S: Into<StringRef>>(&self, section: S) {
        extern "C" {
            fn LLVMSetSection(value: ValueBase, section: *const std::os::raw::c_char);
        }
        let section = section.into();
        let c_str = section.to_cstr();
        unsafe { LLVMSetSection(self.base(), c_str.as_ptr()) }
    }

    fn visibility(&self) -> Visibility {
        extern "C" {
            fn LLVMGetVisibility(value: ValueBase) -> Visibility;
        }
        unsafe { LLVMGetVisibility(self.base()) }
    }

    fn set_visibility(&self, visibility: Visibility) {
        extern "C" {
            fn LLVMSetVisibility(value: ValueBase, visbility: Visibility);
        }
        unsafe { LLVMSetVisibility(self.base(), visibility) }
    }

    fn storage_class(&self) -> DLLStorageClass {
        extern "C" {
            fn LLVMGetDLLStorageClass(value: ValueBase) -> DLLStorageClass;
        }
        unsafe { LLVMGetDLLStorageClass(self.base()) }
    }

    fn set_storage_class(&self, class: DLLStorageClass) {
        extern "C" {
            fn LLVMSetDLLStorageClass(value: ValueBase, class: DLLStorageClass);
        }
        unsafe { LLVMSetDLLStorageClass(self.base(), class) }
    }

    fn unnamed_address(&self) -> UnnamedAddr {
        extern "C" {
            fn LLVMGetUnnamedAddress(value: ValueBase) -> UnnamedAddr;
        }
        unsafe { LLVMGetUnnamedAddress(self.base()) }
    }

    fn set_unnamed_address(&self, addr: UnnamedAddr) {
        extern "C" {
            fn LLVMSetUnnamedAddress(value: ValueBase, addr: UnnamedAddr);
        }
        unsafe { LLVMSetUnnamedAddress(self.base(), addr) }
    }

    /// Returns the "value type" of a global value.
    ///
    /// This is different than the formal type of the value, which is always a pointer type.
    fn global_type(&self) -> TypeBase {
        extern "C" {
            fn LLVMGlobalGetValueType(value: ValueBase) -> TypeBase;
        }
        unsafe { LLVMGlobalGetValueType(self.base()) }
    }
}

/// Thist trait corresponds to the llvm::GlobalObject class and its subtypes
pub trait GlobalObject: GlobalValue {
    fn metadata(&self) -> MetadataEntryIter {
        extern "C" {
            fn LLVMGlobalCopyAllMetadata(
                value: ValueBase,
                len: *mut usize,
            ) -> *const LlvmMetadataEntry;
        }
        let mut len = MaybeUninit::uninit();
        let ptr = unsafe { LLVMGlobalCopyAllMetadata(self.base(), len.as_mut_ptr()) };
        if ptr.is_null() {
            MetadataEntryIter::default()
        } else {
            unsafe { MetadataEntryIter::new(ptr, len.assume_init()) }
        }
    }

    fn set_metadata(&self, kind: MetadataKind, data: Metadata) {
        extern "C" {
            fn LLVMGlobalSetMetadata(value: ValueBase, kind: MetadataKind, data: Metadata);
        }
        unsafe { LLVMGlobalSetMetadata(self.base(), kind, data) }
    }

    fn erase_metadata(&self, kind: MetadataKind) {
        extern "C" {
            fn LLVMGlobalEraseMetadata(value: ValueBase, kind: MetadataKind);
        }
        unsafe { LLVMGlobalEraseMetadata(self.base(), kind) }
    }

    fn clear_metadata(&self) {
        extern "C" {
            fn LLVMGlobalClearMetadata(value: ValueBase);
        }
        unsafe { LLVMGlobalClearMetadata(self.base()) }
    }

    fn get_comdat(&self) -> Comdat {
        extern "C" {
            fn LLVMGetComdat(value: ValueBase) -> Comdat;
        }
        unsafe { LLVMGetComdat(self.base()) }
    }

    fn set_comdat(&self, comdat: Comdat) {
        extern "C" {
            fn LLVMSetComdat(value: ValueBase, comdat: Comdat);
        }
        unsafe { LLVMSetComdat(self.base(), comdat) }
    }
}

/// Represents a variable defined in global scope, e.g. statics, thread-locals, etc.
///
/// GlobalVariables implement llvm::Value, llvm::GlobalValue, and llvm::GlobalObject
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct GlobalVariable(ValueBase);
impl GlobalValue for GlobalVariable {}
impl GlobalObject for GlobalVariable {}
impl Value for GlobalVariable {
    fn base(&self) -> ValueBase {
        self.0
    }
}
impl GlobalVariable {
    pub fn delete(self) {
        extern "C" {
            fn LLVMDeleteGlobal(value: GlobalVariable);
        }
        unsafe { LLVMDeleteGlobal(self) }
    }

    pub fn initializer(self) -> ValueBase {
        extern "C" {
            fn LLVMGetInitializer(value: GlobalVariable) -> ValueBase;
        }
        unsafe { LLVMGetInitializer(self) }
    }

    pub fn set_initializer<V: Value>(self, init: V) {
        extern "C" {
            fn LLVMSetInitializer(value: GlobalVariable, init: ValueBase);
        }
        unsafe { LLVMSetInitializer(self, init.base()) }
    }

    pub fn is_externally_initialized(self) -> bool {
        extern "C" {
            fn LLVMIsExternallyInitialized(value: GlobalVariable) -> bool;
        }
        unsafe { LLVMIsExternallyInitialized(self) }
    }

    pub fn set_externally_initialized(self, is_ext_init: bool) {
        extern "C" {
            fn LLVMSetExternallyInitialized(value: GlobalVariable, is_ext_init: bool);
        }
        unsafe { LLVMSetExternallyInitialized(self, is_ext_init) }
    }

    pub fn is_thread_local(self) -> bool {
        extern "C" {
            fn LLVMIsThreadLocal(value: GlobalVariable) -> bool;
        }
        unsafe { LLVMIsThreadLocal(self) }
    }

    pub fn set_thread_local(self, value: bool) {
        extern "C" {
            fn LLVMSetThreadLocal(value: GlobalVariable, is_thread_local: bool);
        }
        unsafe { LLVMSetThreadLocal(self, value) }
    }

    pub fn thread_local_mode(self) -> ThreadLocalMode {
        extern "C" {
            fn LLVMGetThreadLocalMode(value: GlobalVariable) -> ThreadLocalMode;
        }
        unsafe { LLVMGetThreadLocalMode(self) }
    }

    pub fn set_thread_local_mode(self, mode: ThreadLocalMode) {
        extern "C" {
            fn LLVMSetThreadLocalMode(value: GlobalVariable, mode: ThreadLocalMode);
        }
        unsafe { LLVMSetThreadLocalMode(self, mode) }
    }

    pub fn is_constant(self) -> bool {
        extern "C" {
            fn LLVMIsGlobalConstant(value: GlobalVariable) -> bool;
        }
        unsafe { LLVMIsGlobalConstant(self) }
    }

    pub fn set_constant(self, is_constant: bool) {
        extern "C" {
            fn LLVMSetGlobalConstant(value: GlobalVariable, is_constant: bool);
        }
        unsafe { LLVMSetGlobalConstant(self, is_constant) }
    }
}
impl TryFrom<ValueBase> for GlobalVariable {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::GlobalVariable => Ok(Self(value)),
            _ => Err(InvalidTypeCastError),
        }
    }
}
