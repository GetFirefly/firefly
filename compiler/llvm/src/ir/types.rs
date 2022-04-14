use std::fmt;

use super::*;
use crate::support::StringRef;

extern "C" {
    type LlvmType;
}

/// Represents the kind of a type
///
/// This is primarily used in FFI
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum TypeKind {
    Void = 0,
    FP16,
    FP32,
    FP64,
    FP80,      // 80 bit floating point type (X87)
    FP128,     // 128-bit floating point type (112-bit mantissa)
    FP128_PPC, // 128-bit floating point type (two 64-bits)
    Label,
    Integer, // for any bit width
    Function,
    Struct,
    Array,
    Pointer,
    Vector, // fixed-width SIMD vector type
    Metadata,
    MMX, // x86 mmx
    Token,
    ScalableVector, // scalable SIMD vector type
    BFloat,         // 16-bit brain floating point
    AMX,            // x86 amx
}

/// This trait is implemented by all LLVM types
///
/// Types have the following hierarchy:
///
/// Type:
///   Integers
///   Floats
///   Functions
///   Sequences:
///     Array
///     Pointer
///     Vector
///   Void
///   Label
///   Metadata
pub trait Type {
    /// Returns the kind of type this is
    fn kind(&self) -> TypeKind {
        extern "C" {
            fn LLVMGetTypeKind(ty: TypeBase) -> TypeKind;
        }
        unsafe { LLVMGetTypeKind(self.base()) }
    }

    /// Returns true if this type has a size
    fn is_sized(&self) -> bool {
        extern "C" {
            fn LLVMTypeIsSized(ty: TypeBase) -> bool;
        }
        unsafe { LLVMTypeIsSized(self.base()) }
    }

    /// Returns the context this type was created in
    fn context(&self) -> Context {
        extern "C" {
            fn LLVMGetTypeContext(ty: TypeBase) -> Context;
        }
        unsafe { LLVMGetTypeContext(self.base()) }
    }

    /// Prints a textual representation of this type to stderr
    fn dump(&self) {
        extern "C" {
            fn LLVMDumpType(ty: TypeBase);
        }
        unsafe {
            LLVMDumpType(self.base());
        }
    }

    /// Gets an opaque handle for this type to be used with the FFI bridge
    fn base(&self) -> TypeBase;
}

/// Represents a type that contains one or more elements of a single type
///
/// * arrays
/// * vectors
/// * pointers
pub trait SequentialType: Type {
    /// Returns the element type of this container
    fn element_type(&self) -> TypeBase {
        extern "C" {
            fn LLVMGetElementType(ty: TypeBase) -> TypeBase;
        }
        unsafe { LLVMGetElementType(self.base()) }
    }

    fn subtypes(&self) -> Vec<TypeBase> {
        extern "C" {
            fn LLVMGetSubtypes(ty: TypeBase, results: *mut TypeBase);
        }
        let len = self.arity();
        let mut subtypes = Vec::with_capacity(len);
        unsafe {
            LLVMGetSubtypes(self.base(), subtypes.as_mut_ptr());
            subtypes.set_len(len);
        }
        subtypes
    }

    /// Returns the size of this container, i.e. length
    fn arity(&self) -> usize {
        extern "C" {
            fn LLVMGetNumContainedTypes(ty: TypeBase) -> u32;
        }
        unsafe { LLVMGetNumContainedTypes(self.base()) as usize }
    }
}

/// Represents an opaque handle to an LLVM type for use in the FFI bridge,
/// or in situations where a container of mixed types are needed (e.g. struct fields)
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct TypeBase(*const LlvmType);
impl Type for TypeBase {
    #[inline(always)]
    fn base(&self) -> TypeBase {
        *self
    }
}
impl fmt::Display for TypeBase {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        extern "C" {
            fn LLVMPrintTypeToString(ty: TypeBase) -> *const std::os::raw::c_char;
        }
        let string = unsafe { StringRef::from_ptr(LLVMPrintTypeToString(*self)) };
        write!(f, "{}", &string)
    }
}

macro_rules! impl_type_traits {
    ($ty:ident, $($kind:ident),+) => {
        impl Type for $ty {
            fn base(&self) -> TypeBase {
                self.0
            }
        }
        impl Into<TypeBase> for $ty {
            fn into(self) -> TypeBase {
                self.0
            }
        }
        impl TryFrom<TypeBase> for $ty {
            type Error = InvalidTypeCastError;
            fn try_from(ty: TypeBase) -> Result<Self, Self::Error> {
                match ty.kind() {
                    $(
                        TypeKind::$kind => Ok(Self(ty)),
                    )*
                    _ => Err(InvalidTypeCastError),
                }
            }
        }
        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

/// Represents the void/unit type, i.e. represents nothing
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct VoidType(TypeBase);
impl_type_traits!(VoidType, Void);

/// Represents the set of all floating-point types in LLVM
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct FloatType(TypeBase);
impl_type_traits!(FloatType, FP16, FP32, FP64, FP80, FP128, FP128_PPC, BFloat);

/// Represents the set of all integer types in LLVM
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct IntegerType(TypeBase);
impl_type_traits!(IntegerType, Integer);
impl IntegerType {
    /// Gets the bitwidth of this integer type, e.g. 64
    pub fn bitwidth(self) -> usize {
        extern "C" {
            fn LLVMGetIntTypeWidth(ty: IntegerType) -> u32;
        }
        unsafe { LLVMGetIntTypeWidth(self) as usize }
    }
}

/// Represents the type of a function in LLVM, i.e. its signature
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct FunctionType(TypeBase);
impl_type_traits!(FunctionType, Function);
impl FunctionType {
    pub fn new<R: Type>(return_ty: R, params: &[TypeBase], is_variadic: bool) -> Self {
        extern "C" {
            fn LLVMFunctionType(
                return_ty: TypeBase,
                param_types: *const TypeBase,
                num_params: u32,
                is_variadic: bool,
            ) -> FunctionType;
        }
        unsafe {
            LLVMFunctionType(
                return_ty.base(),
                params.as_ptr(),
                params.len().try_into().unwrap(),
                is_variadic,
            )
        }
    }

    pub fn is_variadic(self) -> bool {
        extern "C" {
            fn LLVMIsFunctionVarArg(ty: FunctionType) -> bool;
        }
        unsafe { LLVMIsFunctionVarArg(self) }
    }

    pub fn return_type(self) -> TypeBase {
        extern "C" {
            fn LLVMGetReturnType(ty: FunctionType) -> TypeBase;
        }
        unsafe { LLVMGetReturnType(self) }
    }

    pub fn arity(self) -> usize {
        extern "C" {
            fn LLVMCountParamTypes(ty: FunctionType) -> u32;
        }
        unsafe { LLVMCountParamTypes(self) as usize }
    }

    pub fn params(self) -> Vec<TypeBase> {
        extern "C" {
            fn LLVMGetParamTypes(ty: FunctionType, params: *mut TypeBase);
        }
        let len = self.arity();
        let mut params = Vec::with_capacity(len);
        unsafe {
            LLVMGetParamTypes(self, params.as_mut_ptr());
            params.set_len(len);
        }
        params
    }
}

/// Represents struct/record types in LLVM
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct StructType(TypeBase);
impl_type_traits!(StructType, Struct);
impl StructType {
    pub fn name(self) -> Option<StringRef> {
        extern "C" {
            fn LLVMGetStructName(ty: StructType) -> *const std::os::raw::c_char;
        }
        let ptr = unsafe { LLVMGetStructName(self) };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { StringRef::from_ptr(ptr) })
        }
    }

    pub fn set_body(self, body: &[TypeBase], packed: bool) {
        extern "C" {
            fn LLVMStructSetBody(
                ty: StructType,
                elements: *const TypeBase,
                num_elements: u32,
                packed: bool,
            );
        }
        unsafe { LLVMStructSetBody(self, body.as_ptr(), body.len().try_into().unwrap(), packed) }
    }

    pub fn arity(self) -> usize {
        extern "C" {
            fn LLVMCountStructElementTypes(ty: StructType) -> u32;
        }
        unsafe { LLVMCountStructElementTypes(self) as usize }
    }

    pub fn element(self, index: usize) -> TypeBase {
        extern "C" {
            fn LLVMStructGetTypeAtIndex(ty: StructType, index: u32) -> TypeBase;
        }
        assert!(
            index < self.arity(),
            "invalid element index, {} is out of bounds",
            index
        );
        unsafe { LLVMStructGetTypeAtIndex(self, index.try_into().unwrap()) }
    }

    pub fn elements(self) -> Vec<TypeBase> {
        extern "C" {
            fn LLVMGetStructElementTypes(ty: StructType, elements: *mut TypeBase);
        }
        let len = self.arity();
        let mut elements = Vec::with_capacity(len);
        unsafe {
            LLVMGetStructElementTypes(self, elements.as_mut_ptr());
            elements.set_len(len);
        }
        elements
    }

    pub fn is_packed(self) -> bool {
        extern "C" {
            fn LLVMIsPackedStruct(ty: StructType) -> bool;
        }
        unsafe { LLVMIsPackedStruct(self) }
    }

    pub fn is_opaque(self) -> bool {
        extern "C" {
            fn LLVMIsOpaqueStruct(ty: StructType) -> bool;
        }
        unsafe { LLVMIsOpaqueStruct(self) }
    }

    pub fn is_literal(self) -> bool {
        extern "C" {
            fn LLVMIsLiteralStruct(ty: StructType) -> bool;
        }
        unsafe { LLVMIsLiteralStruct(self) }
    }
}

/// Represents a fixed size container of a given element type
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ArrayType(TypeBase);
impl_type_traits!(ArrayType, Array);
impl SequentialType for ArrayType {}
impl ArrayType {
    pub fn new<T: Type>(element_ty: T, arity: usize) -> Self {
        extern "C" {
            fn LLVMArrayType(element_ty: TypeBase, arity: u32) -> ArrayType;
        }
        unsafe { LLVMArrayType(element_ty.base(), arity.try_into().unwrap()) }
    }

    pub fn len(self) -> usize {
        extern "C" {
            fn LLVMGetArrayLength(ty: ArrayType) -> u32;
        }
        unsafe { LLVMGetArrayLength(self) as usize }
    }
}

/// Represents a pointer type
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct PointerType(TypeBase);
impl_type_traits!(PointerType, Pointer);
impl SequentialType for PointerType {}
impl PointerType {
    pub fn new<T: Type>(pointee: T, address_space: u32) -> Self {
        extern "C" {
            fn LLVMPointerType(pointee: TypeBase, address_space: u32) -> PointerType;
        }
        unsafe { LLVMPointerType(pointee.base(), address_space) }
    }

    pub fn address_space(self) -> u32 {
        extern "C" {
            fn LLVMGetPointerAddressSpace(ty: PointerType) -> u32;
        }
        unsafe { LLVMGetPointerAddressSpace(self) }
    }
}

/// Represents the type of metadata when used as a value
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct MetadataType(TypeBase);
impl_type_traits!(MetadataType, Metadata);

/// Represents the type of a token value
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct TokenType(TypeBase);
impl_type_traits!(TokenType, Token);

/// Represents the type of a label (e.g. block label) when used as a value
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct LabelType(TypeBase);
impl_type_traits!(LabelType, Label);
