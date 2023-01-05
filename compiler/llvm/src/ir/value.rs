use std::fmt;
use std::mem::MaybeUninit;

use super::*;
use crate::support::StringRef;

extern "C" {
    type LlvmValue;
}

#[repr(C)]
pub enum ValueKind {
    Argument = 0,
    BasicBlock,
    MemoryUse,
    MemoryDef,
    MemoryPhi,
    Function,
    GlobalAlias,
    GlobalIFunc,
    GlobalVariable,
    BlockAddress,
    ConstantExpr,
    ConstantArray,
    ConstantStruct,
    ConstantVector,
    Undef,
    ConstantAggregateZero,
    ConstantDataArray,
    ConstantDataVector,
    ConstantInt,
    ConstantFP,
    ConstantPointerNull,
    ConstantTokenNone,
    MetadataAsValue,
    InlineAsm,
    Instruction,
    Poison,
}

/// This trait is meant to represent the llvm::Value class
///
/// In LLVM, most things inherit from llvm::Value at some point, so this provides a lot of shared
/// functionality
pub trait Value {
    /// Get the string name of this value
    fn name(&self) -> StringRef {
        extern "C" {
            fn LLVMGetValueName2(value: ValueBase, len: *mut usize) -> *const u8;
        }

        unsafe {
            let mut len = MaybeUninit::uninit();
            let ptr = LLVMGetValueName2(self.base(), len.as_mut_ptr());
            assert!(!ptr.is_null());
            StringRef::from_raw_parts(ptr, len.assume_init())
        }
    }

    /// Set the string name of this value
    fn set_name<S: Into<StringRef>>(&self, name: S) {
        extern "C" {
            fn LLVMSetValueName2(value: ValueBase, name: *const u8, len: usize);
        }
        let name = name.into();
        unsafe {
            LLVMSetValueName2(self.base(), name.data, name.len);
        }
    }

    /// Dump a debug representation of this value to stderr
    fn dump(&self) {
        extern "C" {
            fn LLVMDumpValue(value: ValueBase);
        }
        unsafe {
            LLVMDumpValue(self.base());
        }
    }

    /// Get the kind of value this is
    fn kind(&self) -> ValueKind {
        extern "C" {
            fn LLVMGetValueKind(value: ValueBase) -> ValueKind;
        }
        unsafe { LLVMGetValueKind(self.base()) }
    }

    /// Get the type of this value
    fn get_type(&self) -> TypeBase {
        extern "C" {
            fn LLVMTypeOf(value: ValueBase) -> TypeBase;
        }
        unsafe { LLVMTypeOf(self.base()) }
    }

    /// Replace all uses of this value with another one
    fn replace_all_uses_with<V: Value>(&self, other: V) {
        extern "C" {
            fn LLVMReplaceAllUsesWith(old: ValueBase, new: ValueBase);
        }
        unsafe {
            LLVMReplaceAllUsesWith(self.base(), other.base());
        }
    }

    /// Returns true if the underlying ValueBase handle for this value is invalid/null
    #[inline(always)]
    fn is_null(&self) -> bool {
        self.base().is_null()
    }

    /// Returns true if this value represents a null value
    fn is_null_value(&self) -> bool {
        extern "C" {
            fn LLVMIsNull(value: ValueBase) -> bool;
        }
        unsafe { LLVMIsNull(self.base()) }
    }

    /// Returns true if this value is a constant
    fn is_constant(&self) -> bool {
        extern "C" {
            fn LLVMIsConstant(value: ValueBase) -> bool;
        }
        unsafe { LLVMIsConstant(self.base()) }
    }

    /// Returns true if this is a constant string
    fn is_constant_string(&self) -> bool {
        extern "C" {
            fn LLVMIsConstantString(value: ValueBase) -> bool;
        }
        unsafe { LLVMIsConstantString(self.base()) }
    }

    /// Returns true if this value is undef
    fn is_undef(&self) -> bool {
        extern "C" {
            fn LLVMIsUndef(value: ValueBase) -> bool;
        }
        unsafe { LLVMIsUndef(self.base()) }
    }

    /// Returns true if this value is poison
    fn is_poison(&self) -> bool {
        extern "C" {
            fn LLVMIsPoison(value: ValueBase) -> bool;
        }
        unsafe { LLVMIsPoison(self.base()) }
    }

    /// Returns true if this value is an instance of the given type
    fn isa<V: TryFrom<ValueBase>>(&self) -> bool {
        V::try_from(self.base()).is_ok()
    }

    /// Returns an iterator over all the uses of this value
    fn uses(&self) -> ValueUseIter {
        ValueUseIter::new(self.base())
    }

    /// Gets this value as a ValueBase reference
    fn base(&self) -> ValueBase;
}

/// Represents a reference to a value of any llvm::Value subtype
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ValueBase(*const LlvmValue);
impl ValueBase {
    /// Produces a null instance of the llvm::Value class
    ///
    /// Used for optional values in the FFI bridge
    #[inline]
    pub const fn null() -> Self {
        let ptr = unsafe { std::mem::transmute::<*const (), *const LlvmValue>(std::ptr::null()) };
        Self(ptr)
    }
}
impl Value for ValueBase {
    #[inline(always)]
    fn is_null(&self) -> bool {
        self.0.is_null()
    }

    #[inline(always)]
    fn base(&self) -> ValueBase {
        *self
    }
}
impl fmt::Display for ValueBase {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use std::ffi::CStr;

        extern "C" {
            fn LLVMPrintValueToString(value: ValueBase) -> *const std::os::raw::c_char;
        }
        let ptr = unsafe { LLVMPrintValueToString(*self) };
        let c_str = unsafe { CStr::from_ptr(ptr) };
        let s = c_str.to_string_lossy();
        write!(f, "{}", s.as_ref())
    }
}

/// Represents a value of aggregate type (i.e. struct or array, vectors are not considered
/// aggregates)
pub trait Aggregate: Value {}

/// Represents a value of pointer type
pub trait Pointer: Value {}
