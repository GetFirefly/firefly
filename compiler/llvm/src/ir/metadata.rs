use std::mem::MaybeUninit;

use super::*;
use crate::support::*;

extern "C" {
    type LlvmMetadata;
    type LlvmNamedMetadata;
    pub(super) type LlvmMetadataEntry;
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MetadataKind {
    String = 0,
    ConstantAsMetadata,
    LocalAsMetadata,
    DistinctMetadataOperandPlaceholder,
    TupleMetadata,
    DILocation,
    DIExpression,
    DIGlobalVariableExpression,
    DIGenericNode,
    DISubrange,
    DIEnumerator,
    DIBasicType,
    DIDerivedType,
    DICompositeType,
    DISubroutineType,
    DIFile,
    DICompileUnit,
    DISubprogram,
    DILexicalBlock,
    DILexicalBlockFile,
    DINamespace,
    DITemplateTypeParameter,
    DITemplateValueParameter,
    DIGlobalVariable,
    DILocalVariable,
    DILabel,
    DIObjCPropertyMetadata,
    DIImportedEntity,
    DIMacro,
    DIMacroFile,
    DICommonBlock,
    DIStringType,
    DIGenericSubrange,
    DIArgList,
}

#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Metadata(*const LlvmMetadata);
impl Metadata {
    /// Construct a null metadata value
    ///
    /// This is primarily used for optional values in the FFI bridge
    pub const fn null() -> Self {
        let ptr =
            unsafe { std::mem::transmute::<*const (), *const LlvmMetadata>(std::ptr::null()) };
        Self(ptr)
    }

    #[inline(always)]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Return the kind of metadata this is
    pub fn kind(self) -> MetadataKind {
        extern "C" {
            fn LLVMGetMetadataKind(m: Metadata) -> MetadataKind;
        }
        unsafe { LLVMGetMetadataKind(self) }
    }
}

/// Used to represent metadata nodes as values in the IR
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct MetadataValue(ValueBase);
impl Value for MetadataValue {
    fn base(&self) -> ValueBase {
        self.0
    }
}
impl MetadataValue {
    pub fn get(self) -> Metadata {
        extern "C" {
            fn LLVMValueAsMetadata(value: MetadataValue) -> Metadata;
        }
        unsafe { LLVMValueAsMetadata(self) }
    }
}
impl TryFrom<ValueBase> for MetadataValue {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::MetadataAsValue => Ok(Self(value)),
            _ => Err(InvalidTypeCastError),
        }
    }
}

pub(crate) struct NamedMetadataIter(*const LlvmNamedMetadata);
impl NamedMetadataIter {
    pub fn new(module: Module) -> Self {
        extern "C" {
            fn LLVMGetFirstNamedMetadata(module: Module) -> *const LlvmNamedMetadata;
        }
        Self(unsafe { LLVMGetFirstNamedMetadata(module) })
    }
}
impl Iterator for NamedMetadataIter {
    type Item = StringRef;

    fn next(&mut self) -> Option<Self::Item> {
        extern "C" {
            fn LLVMGetNextNamedMetadata(md: *const LlvmNamedMetadata) -> *const LlvmNamedMetadata;
            fn LLVMGetNamedMetadataName(md: *const LlvmNamedMetadata, len: *mut usize)
                -> *const u8;
        }
        if self.0.is_null() {
            return None;
        }
        let next = self.0;
        self.0 = unsafe { LLVMGetNextNamedMetadata(next) };
        let mut len = MaybeUninit::uninit();
        let name = unsafe {
            let ptr = LLVMGetNamedMetadataName(next, len.as_mut_ptr());
            assert!(!ptr.is_null());
            core::slice::from_raw_parts(ptr, len.assume_init()).into()
        };
        Some(name)
    }
}

/// Provides an iterator over metadata entries associated with a value
pub struct MetadataEntryIter {
    ptr: *const LlvmMetadataEntry,
    len: u32,
    pos: u32,
}
impl MetadataEntryIter {
    pub(super) fn new(ptr: *const LlvmMetadataEntry, len: usize) -> Self {
        Self {
            ptr,
            len: len.try_into().unwrap(),
            pos: 0,
        }
    }
}
impl Default for MetadataEntryIter {
    fn default() -> Self {
        Self {
            ptr: unsafe {
                std::mem::transmute::<*const (), *const LlvmMetadataEntry>(std::ptr::null())
            },
            len: 0,
            pos: 0,
        }
    }
}
impl Drop for MetadataEntryIter {
    fn drop(&mut self) {
        extern "C" {
            fn LLVMDisposeValueMetadataEntries(ptr: *const LlvmMetadataEntry);
        }
        if self.ptr.is_null() {
            return;
        }
        unsafe { LLVMDisposeValueMetadataEntries(self.ptr) }
    }
}
impl Iterator for MetadataEntryIter {
    type Item = Metadata;

    fn next(&mut self) -> Option<Self::Item> {
        extern "C" {
            fn LLVMValueMetadataEntriesGetMetadata(
                ptr: *const LlvmMetadataEntry,
                index: u32,
            ) -> Metadata;
        }
        if self.ptr.is_null() || self.pos >= self.len {
            return None;
        }
        let next = self.pos;
        self.pos += 1;
        let metadata = unsafe { LLVMValueMetadataEntriesGetMetadata(self.ptr, next) };
        if metadata.is_null() {
            self.len = 0;
            None
        } else {
            Some(metadata)
        }
    }
}
