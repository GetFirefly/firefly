use std::ffi::c_void;
use std::fmt::{self, Display};
use std::num::NonZeroU32;

use paste::paste;

use crate::support::{self, MlirStringCallback};
use crate::Context;

use super::*;

extern "C" {
    type MlirType;
}

/// This trait is implemented for all concrete type implementations in Rust
pub trait Type {
    /// Returns the context in which this type was created
    fn context(&self) -> Context {
        unsafe { mlir_type_get_context(self.base()) }
    }
    /// Prints the type to the standard error stream
    fn dump(&self) {
        unsafe { mlir_type_dump(self.base()) }
    }
    /// Returns the underlying TypeBase value for this type
    fn base(&self) -> TypeBase;
}

/// Marker trait for integer-like MLIR types, e.g. Integer/IndexType
pub trait IntegerLike: Type {}

/// Represents all MLIR type values in the FFI bridge and provides the foundational
/// API for all concrete type implementations in Rust.
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct TypeBase(*mut MlirType);
impl Type for TypeBase {
    #[inline(always)]
    fn base(&self) -> TypeBase {
        *self
    }
}
impl TypeBase {
    /// Checks whether a type is null
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Returns true if this type is an instance of the concrete type `T`
    #[inline(always)]
    pub fn isa<T>(self) -> bool
    where
        T: TryFrom<TypeBase>,
    {
        T::try_from(self).is_ok()
    }

    /// Tries to convert this type to an instance of the concrete type `T`
    #[inline(always)]
    pub fn dyn_cast<T>(self) -> Result<T, InvalidTypeCastError>
    where
        T: TryFrom<TypeBase, Error = InvalidTypeCastError>,
    {
        T::try_from(self)
    }
}
impl Default for TypeBase {
    fn default() -> Self {
        Self(unsafe { std::mem::transmute::<*mut (), *mut MlirType>(::core::ptr::null_mut()) })
    }
}
impl Display for TypeBase {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe {
            mlir_type_print(
                *self,
                support::write_to_formatter,
                f as *mut _ as *mut c_void,
            );
        }
        Ok(())
    }
}
impl fmt::Pointer for TypeBase {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", self.0)
    }
}
impl Eq for TypeBase {}
impl PartialEq for TypeBase {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlir_type_equal(*self, *other) }
    }
}

extern "C" {
    #[link_name = "mlirTypeGetContext"]
    fn mlir_type_get_context(ty: TypeBase) -> Context;
    #[link_name = "mlirTypeEqual"]
    fn mlir_type_equal(a: TypeBase, b: TypeBase) -> bool;
    #[link_name = "mlirTypePrint"]
    fn mlir_type_print(ty: TypeBase, callback: MlirStringCallback, userdata: *const c_void);
    #[link_name = "mlirTypeDump"]
    fn mlir_type_dump(ty: TypeBase);
}

macro_rules! primitive_builtin_type {
    ($name:ident, $mnemonic:ident) => {
        paste! {
            primitive_builtin_type!($name, $mnemonic, [<$name Type>]);
        }
    };

    ($name:ident, $mnemonic:ident, $ty:ident) => {
        #[repr(transparent)]
        #[derive(Copy, Clone)]
        pub struct $ty(TypeBase);
        impl Type for $ty {
            #[inline]
            fn base(&self) -> TypeBase {
                self.0
            }
        }
        impl $ty {
            pub fn get(context: Context) -> Self {
                paste! {
                    unsafe { [<mlir_ $mnemonic _type_get>](context) }
                }
            }
        }
        impl TryFrom<TypeBase> for $ty {
            type Error = InvalidTypeCastError;

            fn try_from(ty: TypeBase) -> Result<Self, Self::Error> {
                let truth = paste! {
                    unsafe { [<mlir_type_isa_ $mnemonic>](ty) }
                };
                if truth {
                    Ok(Self(ty))
                } else {
                    Err(InvalidTypeCastError)
                }
            }
        }
        impl std::fmt::Display for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "{}", &self.0)
            }
        }
        impl Eq for $ty {}
        impl PartialEq for $ty {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }
        impl PartialEq<TypeBase> for $ty {
            fn eq(&self, other: &TypeBase) -> bool {
                self.0.eq(other)
            }
        }

        paste! {
            primitive_builtin_type!($name, $mnemonic, $ty, [<mlir $ty Get>], [<mlirTypeIsA $name>]);
        }
    };

    ($name:ident, $mnemonic:ident, $ty:ident, $get_name:ident, $isa_name:ident) => {
        extern "C" {
            paste! {
                #[link_name = stringify!($get_name)]
                fn [<mlir_ $mnemonic _type_get>](context: Context) -> $ty;
                #[link_name = stringify!($isa_name)]
                fn [<mlir_type_isa_ $mnemonic>](ty: TypeBase) -> bool;
            }
        }
    };
}

primitive_builtin_type!(None, none);
primitive_builtin_type!(Index, index);
primitive_builtin_type!(Float, f64);

impl IntegerLike for IndexType {}

/// Reprents whether or not a function is variadic (varargs)
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
pub enum Variadic {
    No = 0,
    Yes,
}
impl Into<bool> for Variadic {
    #[inline(always)]
    fn into(self) -> bool {
        self == Variadic::Yes
    }
}

/// Represents the built-in MLIR function type
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct FunctionType(TypeBase);
impl FunctionType {
    pub fn get(context: Context, args: &[TypeBase], results: &[TypeBase]) -> Self {
        let argc = args.len();
        let argv = args.as_ptr();
        let resultc = results.len();
        let resultv = results.as_ptr();
        unsafe { mlir_function_type_get(context, argc, argv, resultc, resultv) }
    }

    pub fn num_inputs(self) -> usize {
        unsafe { mlir_function_type_get_num_inputs(self) }
    }

    pub fn get_input(self, index: usize) -> Option<TypeBase> {
        let ty = unsafe { mlir_function_type_get_input(self, index) };
        if ty.is_null() {
            None
        } else {
            Some(ty)
        }
    }

    pub fn num_results(self) -> usize {
        unsafe { mlir_function_type_get_num_results(self) }
    }

    pub fn get_result(self, index: usize) -> Option<TypeBase> {
        let ty = unsafe { mlir_function_type_get_result(self, index) };
        if ty.is_null() {
            None
        } else {
            Some(ty)
        }
    }
}
impl Type for FunctionType {
    #[inline]
    fn base(&self) -> TypeBase {
        self.0
    }
}
impl TryFrom<TypeBase> for FunctionType {
    type Error = InvalidTypeCastError;

    fn try_from(ty: TypeBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_type_isa_function(ty) } {
            Ok(Self(ty))
        } else {
            Err(InvalidTypeCastError)
        }
    }
}

extern "C" {
    #[link_name = "mlirTypeIsAFunction"]
    fn mlir_type_isa_function(ty: TypeBase) -> bool;
    #[link_name = "mlirFunctionTypeGet"]
    fn mlir_function_type_get(
        context: Context,
        num_args: usize,
        args: *const TypeBase,
        num_results: usize,
        results: *const TypeBase,
    ) -> FunctionType;
    #[link_name = "mlirFunctionTypeGetNumInputs"]
    fn mlir_function_type_get_num_inputs(ty: FunctionType) -> usize;
    #[link_name = "mlirFunctionTypeGetNumResults"]
    fn mlir_function_type_get_num_results(ty: FunctionType) -> usize;
    #[link_name = "mlirFunctionTypeGetInput"]
    fn mlir_function_type_get_input(ty: FunctionType, index: usize) -> TypeBase;
    #[link_name = "mlirFunctionTypeGetResult"]
    fn mlir_function_type_get_result(ty: FunctionType, index: usize) -> TypeBase;
}

/// Represents the set of built-in MLIR integer types
///
/// * May be signed, unsigned, or signless
/// * May be of arbitrary size in bits
///
/// Equality is based on signedness and size in bits
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct IntegerType(TypeBase);
impl IntegerLike for IntegerType {}
impl IntegerType {
    /// Create a signless integer type of the given bit width
    pub fn get(context: Context, bitwidth: usize) -> Self {
        unsafe { mlir_integer_type_get(context, bitwidth as u32) }
    }

    /// Create a signed integer type of the given bit width
    pub fn get_signed(context: Context, bitwidth: usize) -> Self {
        unsafe { mlir_integer_type_signed_get(context, bitwidth as u32) }
    }

    /// Create an unsigned integer type of the given bit width
    pub fn get_unsigned(context: Context, bitwidth: usize) -> Self {
        unsafe { mlir_integer_type_unsigned_get(context, bitwidth as u32) }
    }

    /// Returns the bitwidth of an integer type
    pub fn width(self) -> u32 {
        unsafe { mlir_integer_type_get_width(self) }
    }

    /// Checks whether the given integer type is signless
    pub fn is_signless(self) -> bool {
        unsafe { mlir_integer_type_is_signless(self) }
    }

    /// Checks whether the given integer type is signed.
    pub fn is_signed(self) -> bool {
        unsafe { mlir_integer_type_is_signed(self) }
    }

    /// Checks whether the given integer type is unsigned.
    pub fn is_unsigned(self) -> bool {
        unsafe { mlir_integer_type_is_unsigned(self) }
    }
}
impl Type for IntegerType {
    #[inline]
    fn base(&self) -> TypeBase {
        self.0
    }
}
impl TryFrom<TypeBase> for IntegerType {
    type Error = InvalidTypeCastError;

    fn try_from(ty: TypeBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_type_isa_integer(ty) } {
            Ok(Self(ty))
        } else {
            Err(InvalidTypeCastError)
        }
    }
}

extern "C" {
    #[link_name = "mlirTypeIsAInteger"]
    fn mlir_type_isa_integer(ty: TypeBase) -> bool;
    #[link_name = "mlirIntegerTypeGet"]
    fn mlir_integer_type_get(context: Context, width: u32) -> IntegerType;
    #[link_name = "mlirIntegerTypeSignedGet"]
    fn mlir_integer_type_signed_get(context: Context, width: u32) -> IntegerType;
    #[link_name = "mlirIntegerTypeUnsignedGet"]
    fn mlir_integer_type_unsigned_get(context: Context, width: u32) -> IntegerType;
    #[link_name = "mlirIntegerTypeGetWidth"]
    fn mlir_integer_type_get_width(ty: IntegerType) -> u32;
    #[link_name = "mlirIntegerTypeIsSignless"]
    fn mlir_integer_type_is_signless(ty: IntegerType) -> bool;
    #[link_name = "mlirIntegerTypeIsSigned"]
    fn mlir_integer_type_is_signed(ty: IntegerType) -> bool;
    #[link_name = "mlirIntegerTypeIsUnsigned"]
    fn mlir_integer_type_is_unsigned(ty: IntegerType) -> bool;
}

/// Represents the built-in MLIR tuple type.
///
/// Tuples can be of arbitrary size and shape, i.e. contain heterogenous elements.
///
/// MLIR doesn't define operations on tuples, so that is up to custom dialects to handle.
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct TupleType(TypeBase);
impl TupleType {
    pub fn get(context: Context, elements: &[TypeBase]) -> Self {
        unsafe { mlir_tuple_type_get(context, elements.len(), elements.as_ptr()) }
    }

    pub fn arity(&self) -> usize {
        unsafe { mlir_tuple_type_get_num_types(self.0) }
    }

    pub fn get_element_type(&self, index: usize) -> TypeBase {
        unsafe { mlir_tuple_type_get_type(self.0, index) }
    }
}
impl Type for TupleType {
    #[inline]
    fn base(&self) -> TypeBase {
        self.0
    }
}
impl TryFrom<TypeBase> for TupleType {
    type Error = InvalidTypeCastError;

    fn try_from(ty: TypeBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_type_isa_tuple(ty) } {
            Ok(Self(ty))
        } else {
            Err(InvalidTypeCastError)
        }
    }
}

extern "C" {
    #[link_name = "mlirTypeIsATuple"]
    fn mlir_type_isa_tuple(ty: TypeBase) -> bool;
    #[link_name = "mlirTupleTypeGet"]
    fn mlir_tuple_type_get(
        context: Context,
        num_elements: usize,
        elements: *const TypeBase,
    ) -> TupleType;
    #[link_name = "mlirTupleTypeGetNumTypes"]
    fn mlir_tuple_type_get_num_types(ty: TypeBase) -> usize;
    #[link_name = "mlirTupleTypeGetType"]
    fn mlir_tuple_type_get_type(ty: TypeBase, index: usize) -> TypeBase;
}

/// Represents the address space of a pointer in LLVM
///
/// In general, pointers are created in the default address space,
/// but when representing pointers that are special in some way, such
/// as garbage-collected references, non-standard address spaces are
/// used instead.
#[repr(u32)]
pub enum AddressSpace {
    Default = 0,
    Other(NonZeroU32),
}
impl Default for AddressSpace {
    #[inline(always)]
    fn default() -> Self {
        Self::Default
    }
}
impl Into<u32> for AddressSpace {
    fn into(self) -> u32 {
        match self {
            Self::Default => 0,
            Self::Other(value) => value.get(),
        }
    }
}
impl Into<i64> for AddressSpace {
    fn into(self) -> i64 {
        match self {
            Self::Default => 0,
            Self::Other(value) => value.get().into(),
        }
    }
}
impl AddressSpace {
    pub const fn new(addrspace: u32) -> Self {
        match addrspace {
            0 => Self::Default,
            n => Self::Other(unsafe { NonZeroU32::new_unchecked(n) }),
        }
    }
}

/// Represents the built-in MLIR MemRef and UnrankedMemRef types
///
/// A MemRef is a type representing a region of memory of a specific size and shape.
/// The layout can be dynamic or static, of unknown, single or multi-dimensional rank, contiguous or sparse.
///
/// MemRefs have a specific fixed element type, i.e. they aren't meant to represent things like
/// heterogenous tuples, but rather vectors/tensors/arrays of a specific type. In MLIR they are generally
/// used with numeric types to solve for common ML tasks.
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct MemRefType(TypeBase);
impl MemRefType {
    /// Creates a MemRef type with the given element type, rank, shape, and address space, with
    /// a potentially empty list of affine maps.
    pub fn get<T: Type>(
        element_type: T,
        shape: &[u64],
        layout: AffineMapAttr,
        memory_space: AddressSpace,
    ) -> Self {
        let memory_space = Self::get_memory_space_attr(element_type.context(), memory_space);
        unsafe {
            mlir_memref_type_get(
                element_type.base(),
                shape.len(),
                shape.as_ptr(),
                layout.base(),
                memory_space,
            )
        }
    }

    /// Creates a MemRef of the given element type, shape and address space.
    ///
    /// This type is different than the one returned by `get`, as it has no affine maps, i.e.
    /// it represents a default row-major contiguous memref.
    pub fn get_contiguous<T: Type>(
        element_type: T,
        shape: &[u64],
        memory_space: AddressSpace,
    ) -> Self {
        let memory_space = Self::get_memory_space_attr(element_type.context(), memory_space);
        unsafe {
            mlir_memref_type_contiguous_get(
                element_type.base(),
                shape.len(),
                shape.as_ptr(),
                memory_space,
            )
        }
    }

    /// Creates a MemRef of the given element type, with dynamic rank, in the provided address space
    pub fn get_unranked<T: Type>(element_type: T, memory_space: AddressSpace) -> Self {
        let memory_space = Self::get_memory_space_attr(element_type.context(), memory_space);
        unsafe { mlir_unranked_memref_type_get(element_type.base(), memory_space) }
    }

    /// Returns true if this MemRef is unranked
    pub fn is_unranked(&self) -> bool {
        unsafe { mlir_type_isa_unranked_memref(self.0) }
    }

    /// Returns an attribute that implements MemRefLayoutAttrInterface, typically an AffineMapAttr, but not always
    pub fn layout(self) -> AttributeBase {
        unsafe { mlir_memref_type_get_layout(self) }
    }

    /// Returns the affine map determined from the layout
    pub fn affine_map(self) -> AffineMap {
        unsafe { mlir_memref_type_get_affine_map(self) }
    }

    /// Returns the address space in which this memref is allocated
    pub fn memory_space(self) -> AddressSpace {
        let attr = unsafe { mlir_memref_type_get_memory_space(self) };
        let value = IntegerAttr::try_from(attr).unwrap().value();
        match value {
            0 => AddressSpace::Default,
            n => AddressSpace::Other(unsafe {
                NonZeroU32::new_unchecked(
                    n.try_into().expect("invalid address space integral value"),
                )
            }),
        }
    }

    fn get_memory_space_attr(context: Context, memory_space: AddressSpace) -> IntegerAttr {
        let i64_ty = IntegerType::get(context, 64);
        IntegerAttr::get(i64_ty, memory_space.into()).into()
    }
}
impl Type for MemRefType {
    #[inline]
    fn base(&self) -> TypeBase {
        self.0
    }
}
impl TryFrom<TypeBase> for MemRefType {
    type Error = InvalidTypeCastError;

    fn try_from(ty: TypeBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_type_isa_memref(ty) || mlir_type_isa_unranked_memref(ty) } {
            Ok(Self(ty))
        } else {
            Err(InvalidTypeCastError)
        }
    }
}

extern "C" {
    #[link_name = "mlirTypeIsAMemRef"]
    fn mlir_type_isa_memref(ty: TypeBase) -> bool;
    #[link_name = "mlirTypeIsaUnrankedMemRef"]
    fn mlir_type_isa_unranked_memref(ty: TypeBase) -> bool;
    #[link_name = "mlirMemRefTypeGet"]
    fn mlir_memref_type_get(
        element_type: TypeBase,
        rank: usize,
        shape: *const u64,
        layout: AttributeBase,
        memory_space: IntegerAttr,
    ) -> MemRefType;
    #[link_name = "mlirMemRefTypeContiguousGet"]
    fn mlir_memref_type_contiguous_get(
        element_type: TypeBase,
        rank: usize,
        shape: *const u64,
        memory_space: IntegerAttr,
    ) -> MemRefType;
    #[link_name = "mlirUnrankedMemRefTypeGet"]
    fn mlir_unranked_memref_type_get(
        element_type: TypeBase,
        memory_space: IntegerAttr,
    ) -> MemRefType;
    #[link_name = "mlirMemRefTypeGetLayout"]
    fn mlir_memref_type_get_layout(ty: MemRefType) -> AttributeBase;
    #[link_name = "mlirMemRefTypeGetAffineMap"]
    fn mlir_memref_type_get_affine_map(ty: MemRefType) -> AffineMap;
    #[link_name = "mlirMemRefTypeGetMemorySpace"]
    fn mlir_memref_type_get_memory_space(ty: MemRefType) -> AttributeBase;
}
