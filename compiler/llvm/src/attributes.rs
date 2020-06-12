use std::ffi::CString;

use crate::{Type, Value};

#[derive(Copy, Clone)]
pub enum AttributePlace {
    ReturnValue,
    Argument(u32),
    Function,
}

impl AttributePlace {
    pub fn as_uint(self) -> libc::c_uint {
        match self {
            AttributePlace::ReturnValue => 0,
            AttributePlace::Argument(i) => 1 + i,
            AttributePlace::Function => !0,
        }
    }
}

/// Matches lumen::Attribute::AttrKind in Attributes.cpp
/// Semantically a subset of the C++ enum llvm::Attribute::AttrKind,
/// though it is not ABI compatible (since it's a C++ enum)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub enum Attribute {
    AlwaysInline = 0,
    ByVal = 1,
    Cold = 2,
    InlineHint = 3,
    MinSize = 4,
    Naked = 5,
    NoAlias = 6,
    NoCapture = 7,
    NoInline = 8,
    NonNull = 9,
    NoRedZone = 10,
    NoReturn = 11,
    NoUnwind = 12,
    OptimizeForSize = 13,
    ReadOnly = 14,
    ArgMemOnly = 15,
    SExt = 16,
    StructRet = 17,
    UWTable = 18,
    ZExt = 19,
    InReg = 20,
    SanitizeThread = 21,
    SanitizeAddress = 22,
    SanitizeMemory = 23,
    NonLazyBind = 24,
    OptimizeNone = 25,
    ReturnsTwice = 26,
}

extern "C" {
    pub fn LLVMLumenAddCallSiteAttribute(instr: Value, index: libc::c_uint, attr: Attribute);
    pub fn LLVMLumenAddAlignmentCallSiteAttr(instr: Value, index: libc::c_uint, bytes: u32);
    pub fn LLVMLumenAddDereferenceableCallSiteAttr(instr: Value, index: libc::c_uint, bytes: u64);
    pub fn LLVMLumenAddDereferenceableOrNullCallSiteAttr(
        instr: Value,
        index: libc::c_uint,
        bytes: u64,
    );
    pub fn LLVMLumenAddByValCallSiteAttr(instr: Value, index: libc::c_uint, ty: Type);

    pub fn LLVMLumenAddAlignmentAttr(fun: Value, index: libc::c_uint, bytes: u32);
    pub fn LLVMLumenAddDereferenceableAttr(fun: Value, index: libc::c_uint, bytes: u64);
    pub fn LLVMLumenAddDereferenceableOrNullAttr(fun: Value, index: libc::c_uint, bytes: u64);
    pub fn LLVMLumenAddByValAttr(fun: Value, index: libc::c_uint, ty: Type);
    pub fn LLVMLumenAddFunctionAttribute(fun: Value, index: libc::c_uint, attr: Attribute);
    pub fn LLVMLumenAddFunctionAttrStringValue(
        fun: Value,
        index: libc::c_uint,
        name: *const libc::c_char,
        value: *const libc::c_char,
    );
    pub fn LLVMLumenRemoveFunctionAttributes(fun: Value, index: libc::c_uint, attr: Attribute);
}
