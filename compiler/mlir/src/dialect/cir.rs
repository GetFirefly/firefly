use std::ops::Deref;

use liblumen_binary::Endianness;
use liblumen_intern::Symbol;
use paste::paste;

use crate::dialect::llvm;
use crate::ir::*;
use crate::support::StringRef;

/// Primary builder for the CIR dialect
///
/// Wraps mlir::OpBuilder and provides functionality for constructing dialect operations, types, and attributes
#[derive(Copy, Clone)]
pub struct CirBuilder<'a, B: OpBuilder> {
    builder: &'a B,
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    pub fn new(builder: &'a B) -> Self {
        Self { builder }
    }

    /// NOTE: The field types given must be types from the LLVM dialect
    pub fn get_struct_type(&self, fields: &[TypeBase]) -> llvm::StructType {
        llvm::StructType::get(self.context(), fields)
    }
}
impl<'a, B: OpBuilder> Builder for CirBuilder<'a, B> {
    #[inline]
    fn base(&self) -> BuilderBase {
        self.builder.base()
    }
}
impl<'a, B: OpBuilder> OpBuilder for CirBuilder<'a, B> {}
impl<'a, B: OpBuilder> Deref for CirBuilder<'a, B> {
    type Target = B;

    fn deref(&self) -> &Self::Target {
        self.builder
    }
}

//----------------------------
// Attributes
//----------------------------

/// AtomAttr is used to represent atom literals in MLIR
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct AtomAttr(AttributeBase);
impl AtomAttr {
    #[inline]
    pub fn get<A: Into<AtomRef>, T: AtomLike>(value: A, ty: T) -> Self {
        unsafe { mlir_cir_atom_attr_get(value.into(), ty.base()) }
    }

    #[inline]
    pub fn value(&self) -> AtomRef {
        unsafe { mlir_cir_atom_attr_valueof(self.0) }
    }
}
impl Attribute for AtomAttr {
    #[inline]
    fn base(&self) -> AttributeBase {
        self.0
    }
}
impl TryFrom<AttributeBase> for AtomAttr {
    type Error = ();

    #[inline]
    fn try_from(attr: AttributeBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_cir_atom_attr_isa(attr) } {
            Ok(Self(attr))
        } else {
            Err(())
        }
    }
}
impl ::std::fmt::Debug for AtomAttr {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "AtomAttr({:p})", &self.0)
    }
}
impl ::std::fmt::Display for AtomAttr {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "{}", self.base())
    }
}

impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn get_atom_attr<A: Into<AtomRef>, T: AtomLike>(&self, value: A, ty: T) -> AtomAttr {
        AtomAttr::get(value, ty)
    }
}

extern "C" {
    #[link_name = "mlirCirAtomAttrGet"]
    fn mlir_cir_atom_attr_get(value: AtomRef, ty: TypeBase) -> AtomAttr;
    #[link_name = "mlirCirAtomAttrIsA"]
    fn mlir_cir_atom_attr_isa(attr: AttributeBase) -> bool;
    #[link_name = "mlirCirAtomAttrValueOf"]
    fn mlir_cir_atom_attr_valueof(attr: AttributeBase) -> AtomRef;
}

/// EndiannessAttr is used to represent constant endianness values in MLIR
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct EndiannessAttr(AttributeBase);
impl EndiannessAttr {
    #[inline]
    pub fn get(context: Context, value: Endianness) -> Self {
        unsafe { mlir_cir_endianness_attr_get(value.into(), context) }
    }

    #[inline]
    pub fn value(&self) -> Endianness {
        unsafe { mlir_cir_endianness_attr_valueof(self.0) }
    }
}
impl Attribute for EndiannessAttr {
    #[inline]
    fn base(&self) -> AttributeBase {
        self.0
    }
}
impl TryFrom<AttributeBase> for EndiannessAttr {
    type Error = ();

    #[inline]
    fn try_from(attr: AttributeBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_cir_endianness_attr_isa(attr) } {
            Ok(Self(attr))
        } else {
            Err(())
        }
    }
}
impl ::std::fmt::Debug for EndiannessAttr {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "EndiannessAttr({:p})", &self.0)
    }
}
impl ::std::fmt::Display for EndiannessAttr {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "{}", self.base())
    }
}

impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn get_endianness_attr(&self, value: Endianness) -> EndiannessAttr {
        EndiannessAttr::get(self.context(), value)
    }
}

extern "C" {
    #[link_name = "mlirCirEndiannessAttrGet"]
    fn mlir_cir_endianness_attr_get(value: Endianness, context: Context) -> EndiannessAttr;
    #[link_name = "mlirCirEndiannessAttrIsA"]
    fn mlir_cir_endianness_attr_isa(attr: AttributeBase) -> bool;
    #[link_name = "mlirCirEndiannessAttrValueOf"]
    fn mlir_cir_endianness_attr_valueof(attr: AttributeBase) -> Endianness;
}

/// NoneAttr is used to represent a constant none value
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct NoneAttr(AttributeBase);
impl NoneAttr {
    #[inline]
    pub fn get(context: Context) -> Self {
        unsafe { mlir_cir_none_attr_get(context) }
    }
}
impl Attribute for NoneAttr {
    #[inline]
    fn base(&self) -> AttributeBase {
        self.0
    }
}
impl TryFrom<AttributeBase> for NoneAttr {
    type Error = ();

    #[inline]
    fn try_from(attr: AttributeBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_cir_none_attr_isa(attr) } {
            Ok(Self(attr))
        } else {
            Err(())
        }
    }
}
impl ::std::fmt::Debug for NoneAttr {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "NoneAttr({:p})", &self.0)
    }
}
impl ::std::fmt::Display for NoneAttr {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "{}", self.base())
    }
}

impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn get_none_attr(&self) -> NoneAttr {
        NoneAttr::get(self.context())
    }
}

extern "C" {
    #[link_name = "mlirCirNoneAttrGet"]
    fn mlir_cir_none_attr_get(context: Context) -> NoneAttr;
    #[link_name = "mlirCirNoneAttrIsA"]
    fn mlir_cir_none_attr_isa(attr: AttributeBase) -> bool;
}

/// NilAttr is used to represent a constant nil value
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct NilAttr(AttributeBase);
impl NilAttr {
    #[inline]
    pub fn get(context: Context) -> Self {
        unsafe { mlir_cir_nil_attr_get(context) }
    }
}
impl Attribute for NilAttr {
    #[inline]
    fn base(&self) -> AttributeBase {
        self.0
    }
}
impl TryFrom<AttributeBase> for NilAttr {
    type Error = ();

    #[inline]
    fn try_from(attr: AttributeBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_cir_nil_attr_isa(attr) } {
            Ok(Self(attr))
        } else {
            Err(())
        }
    }
}
impl ::std::fmt::Debug for NilAttr {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "NilAttr({:p})", &self.0)
    }
}
impl ::std::fmt::Display for NilAttr {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "{}", self.base())
    }
}

impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn get_nil_attr(&self) -> NilAttr {
        NilAttr::get(self.context())
    }
}

extern "C" {
    #[link_name = "mlirCirNilAttrGet"]
    fn mlir_cir_nil_attr_get(context: Context) -> NilAttr;
    #[link_name = "mlirCirNilAttrIsA"]
    fn mlir_cir_nil_attr_isa(attr: AttributeBase) -> bool;
}

/// BoolAttr is used to represent a constant boolean atom value
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct BoolAttr(AttributeBase);
impl BoolAttr {
    #[inline]
    pub fn get(context: Context, value: bool) -> Self {
        unsafe { mlir_cir_bool_attr_get(context, value) }
    }

    #[inline]
    pub fn value(&self) -> bool {
        unsafe { mlir_cir_bool_attr_value_of(self.base()) }
    }
}
impl Attribute for BoolAttr {
    #[inline]
    fn base(&self) -> AttributeBase {
        self.0
    }
}
impl TryFrom<AttributeBase> for BoolAttr {
    type Error = ();

    #[inline]
    fn try_from(attr: AttributeBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_cir_bool_attr_isa(attr) } {
            Ok(Self(attr))
        } else {
            Err(())
        }
    }
}
impl ::std::fmt::Debug for BoolAttr {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "BoolAttr({:p})", &self.0)
    }
}
impl ::std::fmt::Display for BoolAttr {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "{}", self.base())
    }
}

impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn get_cir_bool_attr(&self, value: bool) -> BoolAttr {
        BoolAttr::get(self.context(), value)
    }
}

extern "C" {
    #[link_name = "mlirCirBoolAttrGet"]
    fn mlir_cir_bool_attr_get(contxt: Context, value: bool) -> BoolAttr;
    #[link_name = "mlirCirBoolAttrIsA"]
    fn mlir_cir_bool_attr_isa(attr: AttributeBase) -> bool;
    #[link_name = "mlirCirBoolAttrValueOf"]
    fn mlir_cir_bool_attr_value_of(attr: AttributeBase) -> bool;
}

/// IsizeAttr is used to represent a constant isize value
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct IsizeAttr(AttributeBase);
impl IsizeAttr {
    #[inline]
    pub fn get(context: Context, value: u64) -> Self {
        unsafe { mlir_cir_isize_attr_get(context, value) }
    }

    #[inline]
    pub fn value(&self) -> u64 {
        unsafe { mlir_cir_isize_attr_value_of(self.base()) }
    }
}
impl Attribute for IsizeAttr {
    #[inline]
    fn base(&self) -> AttributeBase {
        self.0
    }
}
impl TryFrom<AttributeBase> for IsizeAttr {
    type Error = ();

    #[inline]
    fn try_from(attr: AttributeBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_cir_isize_attr_isa(attr) } {
            Ok(Self(attr))
        } else {
            Err(())
        }
    }
}
impl ::std::fmt::Debug for IsizeAttr {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "IsizeAttr({:p})", &self.0)
    }
}
impl ::std::fmt::Display for IsizeAttr {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "{}", self.base())
    }
}

impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn get_isize_attr(&self, value: u64) -> IsizeAttr {
        IsizeAttr::get(self.context(), value)
    }
}

extern "C" {
    #[link_name = "mlirCirIsizeAttrGet"]
    fn mlir_cir_isize_attr_get(contxt: Context, value: u64) -> IsizeAttr;
    #[link_name = "mlirCirIsizeAttrIsA"]
    fn mlir_cir_isize_attr_isa(attr: AttributeBase) -> bool;
    #[link_name = "mlirCirIsizeAttrValueOf"]
    fn mlir_cir_isize_attr_value_of(attr: AttributeBase) -> u64;
}

/// FloatAttr is used to represent a constant floating-point value
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct FloatAttr(AttributeBase);
impl FloatAttr {
    #[inline]
    pub fn get(context: Context, value: f64) -> Self {
        unsafe { mlir_cir_float_attr_get(context, value) }
    }

    #[inline]
    pub fn value(&self) -> f64 {
        unsafe { mlir_cir_float_attr_value_of(self.base()) }
    }
}
impl Attribute for FloatAttr {
    #[inline]
    fn base(&self) -> AttributeBase {
        self.0
    }
}
impl TryFrom<AttributeBase> for FloatAttr {
    type Error = ();

    #[inline]
    fn try_from(attr: AttributeBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_cir_float_attr_isa(attr) } {
            Ok(Self(attr))
        } else {
            Err(())
        }
    }
}
impl ::std::fmt::Debug for FloatAttr {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "FloatAttr({:p})", &self.0)
    }
}
impl ::std::fmt::Display for FloatAttr {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "{}", self.base())
    }
}

impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn get_float_attr(&self, value: f64) -> FloatAttr {
        FloatAttr::get(self.context(), value)
    }
}

extern "C" {
    #[link_name = "mlirCirFloatAttrGet"]
    fn mlir_cir_float_attr_get(contxt: Context, value: f64) -> FloatAttr;
    #[link_name = "mlirCirFloatAttrIsA"]
    fn mlir_cir_float_attr_isa(attr: AttributeBase) -> bool;
    #[link_name = "mlirCirFloatAttrValueOf"]
    fn mlir_cir_float_attr_value_of(attr: AttributeBase) -> f64;
}

//----------------------------
// Types
//----------------------------

/// Marker type for CIR types
pub trait CirType: Type {}

/// Tuples from the standard dialect are permitted in CIR
impl CirType for crate::ir::TupleType {}
/// MemRefs from the standard dialect are permitted in CIR
impl CirType for crate::ir::MemRefType {}

/// Marker type for atom-like types
pub trait AtomLike: CirType {}

macro_rules! primitive_cir_type {
    ($ty:ident, $name:ident) => {
        #[repr(transparent)]
        #[derive(Copy, Clone)]
        pub struct $ty(TypeBase);
        impl Type for $ty {
            #[inline]
            fn base(&self) -> TypeBase {
                self.0
            }
        }
        impl CirType for $ty {}
        impl $ty {
            pub fn get(context: Context) -> Self {
                paste! {
                    unsafe { [<mlir_cir_type_get_ $name>](context) }
                }
            }
        }
        impl TryFrom<TypeBase> for $ty {
            type Error = ();

            fn try_from(ty: TypeBase) -> Result<Self, Self::Error> {
                let truth = paste! {
                    unsafe { [<mlir_cir_type_isa_ $name>](ty) }
                };
                if truth {
                    Ok(Self(ty))
                } else {
                    Err(())
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
            impl<'a, B: OpBuilder> CirBuilder<'a, B> {
                #[inline]
                pub fn [<get_cir_ $name _type>](&self) -> $ty {
                    $ty::get(self.context())
                }
            }
        }

        paste! {
            primitive_cir_type!($ty, $name, [<mlirCir $ty Get>], [<mlirCirIsA $ty>]);
        }
    };

    ($ty:ty, $name:ident, $get_name:ident, $isa_name:ident) => {
        extern "C" {
            paste! {
                #[link_name = stringify!($get_name)]
                fn [<mlir_cir_type_get_ $name>](context: Context) -> $ty;
                #[link_name = stringify!($isa_name)]
                fn [<mlir_cir_type_isa_ $name>](ty: TypeBase) -> bool;
            }
        }
    };
}

macro_rules! generic_type {
    ($ty:ident, $name:ident) => {
        #[repr(transparent)]
        #[derive(Copy, Clone)]
        pub struct $ty<T: Type> {
            ty: TypeBase,
            _marker: core::marker::PhantomData<T>,
        }
        impl<T: Type> Type for $ty<T> {
            #[inline]
            fn base(&self) -> TypeBase {
                self.ty
            }
        }
        impl<T: Type> $ty<T> {
            pub fn get(ty: T) -> Self {
                let ty = paste! {
                    unsafe { [<mlir_cir_type_get_ $name>](ty.base()) }
                };
                Self {
                    ty,
                    _marker: core::marker::PhantomData,
                }
            }
        }
        impl<T: Type> TryFrom<TypeBase> for $ty<T> {
            type Error = ();

            fn try_from(ty: TypeBase) -> Result<Self, Self::Error> {
                let truth = paste! {
                    unsafe { [<mlir_cir_type_isa_ $name>](ty) }
                };
                if truth {
                    Ok(Self {
                        ty,
                        _marker: core::marker::PhantomData,
                    })
                } else {
                    Err(())
                }
            }
        }
        impl<T: Type> std::fmt::Display for $ty<T> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "{}", &self.ty)
            }
        }
        impl<T: Type> Eq for $ty<T> {}
        impl<T: Type> PartialEq for $ty<T> {
            fn eq(&self, other: &Self) -> bool {
                self.ty == other.ty
            }
        }
        impl<T: Type> PartialEq<TypeBase> for $ty<T> {
            fn eq(&self, other: &TypeBase) -> bool {
                self.ty.eq(other)
            }
        }

        paste! {
            impl<'a, B: OpBuilder> CirBuilder<'a, B> {
                #[inline]
                pub fn [<get_cir_ $name _type>]<T: Type>(&self, element: T) -> $ty<T> {
                    $ty::get(element)
                }
            }
        }

        paste! {
            generic_type!($ty, $name, [<mlirCir $ty Get>], [<mlirCirIsA $ty>]);
        }
    };

    ($ty:ty, $name:ident, $get_name:ident, $isa_name:ident) => {
        extern "C" {
            paste! {
                #[link_name = stringify!($get_name)]
                fn [<mlir_cir_type_get_ $name>](element: TypeBase) -> TypeBase;
                #[link_name = stringify!($isa_name)]
                fn [<mlir_cir_type_isa_ $name>](ty: TypeBase) -> bool;
            }
        }
    };
}

macro_rules! generic_cir_type {
    ($ty:ident, $name:ident) => {
        #[repr(transparent)]
        #[derive(Copy, Clone)]
        pub struct $ty<T: CirType> {
            ty: TypeBase,
            _marker: core::marker::PhantomData<T>,
        }
        impl<T: CirType> Type for $ty<T> {
            #[inline]
            fn base(&self) -> TypeBase {
                self.ty
            }
        }
        impl<T: CirType> CirType for $ty<T> {}
        impl<T: CirType> $ty<T> {
            pub fn get(ty: T) -> Self {
                let ty = paste! {
                    unsafe { [<mlir_cir_type_get_ $name>](ty.base()) }
                };
                Self {
                    ty,
                    _marker: core::marker::PhantomData,
                }
            }
        }
        impl<T: CirType> TryFrom<TypeBase> for $ty<T> {
            type Error = ();

            fn try_from(ty: TypeBase) -> Result<Self, Self::Error> {
                let truth = paste! {
                    unsafe { [<mlir_cir_type_isa_ $name>](ty) }
                };
                if truth {
                    Ok(Self {
                        ty,
                        _marker: core::marker::PhantomData,
                    })
                } else {
                    Err(())
                }
            }
        }
        impl<T: CirType> std::fmt::Display for $ty<T> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "{}", &self.ty)
            }
        }
        impl<T: CirType> Eq for $ty<T> {}
        impl<T: CirType> PartialEq for $ty<T> {
            fn eq(&self, other: &Self) -> bool {
                self.ty == other.ty
            }
        }
        impl<T: CirType> PartialEq<TypeBase> for $ty<T> {
            fn eq(&self, other: &TypeBase) -> bool {
                self.ty.eq(other)
            }
        }

        paste! {
            impl<'a, B: OpBuilder> CirBuilder<'a, B> {
                #[inline]
                pub fn [<get_cir_ $name _type>]<T: CirType>(&self, element: T) -> $ty<T> {
                    $ty::get(element)
                }
            }
        }

        paste! {
            generic_cir_type!($ty, $name, [<mlirCir $ty Get>], [<mlirCirIsA $ty>]);
        }
    };

    ($ty:ty, $name:ident, $get_name:ident, $isa_name:ident) => {
        extern "C" {
            paste! {
                #[link_name = stringify!($get_name)]
                fn [<mlir_cir_type_get_ $name>](element: TypeBase) -> TypeBase;
                #[link_name = stringify!($isa_name)]
                fn [<mlir_cir_type_isa_ $name>](ty: TypeBase) -> bool;
            }
        }
    };
}

macro_rules! container_cir_type {
    ($ty:ident, $name:ident) => {
        #[repr(transparent)]
        #[derive(Copy, Clone)]
        pub struct $ty(TypeBase);
        impl Type for $ty {
            #[inline]
            fn base(&self) -> TypeBase {
                self.0
            }
        }
        impl CirType for $ty {}
        impl $ty {
            pub fn get(context: Context, elements: &[TypeBase]) -> Self {
                paste! {
                    unsafe { [<mlir_cir_type_get_ $name>](context, elements.len(), elements.as_ptr()) }
                }
            }
        }
        impl TryFrom<TypeBase> for $ty {
            type Error = ();

            fn try_from(ty: TypeBase) -> Result<Self, Self::Error> {
                let truth = paste! {
                    unsafe { [<mlir_cir_type_isa_ $name>](ty) }
                };
                if truth {
                    Ok(Self(ty))
                } else {
                    Err(())
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
            fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
        }
        impl PartialEq<TypeBase> for $ty {
            fn eq(&self, other: &TypeBase) -> bool { self.0.eq(other) }
        }

        paste! {
            impl<'a, B: OpBuilder> CirBuilder<'a, B> {
                #[inline]
                pub fn [<get_cir_ $name _type>](&self, elements: &[TypeBase]) -> $ty {
                    $ty::get(self.context(), elements)
                }
            }
        }

        paste! {
            container_cir_type!($ty, $name, [<mlirCir $ty Get>], [<mlirCirIsA $ty>]);
        }
    };

    ($ty:ident, $name:ident, $get_name:ident, $isa_name:ident) => {
        extern "C" {
            paste! {
                #[link_name = stringify!($get_name)]
                fn [<mlir_cir_type_get_ $name>](context: Context, num_elements: usize, elements: *const TypeBase) -> $ty;
                #[link_name = stringify!($isa_name)]
                fn [<mlir_cir_type_isa_ $name>](ty: TypeBase) -> bool;
            }
        }
    }
}

primitive_cir_type!(NoneType, none);
primitive_cir_type!(TermType, term);
primitive_cir_type!(NumberType, number);
primitive_cir_type!(IntegerType, integer);
primitive_cir_type!(FloatType, float);
primitive_cir_type!(AtomType, atom);
primitive_cir_type!(BoolType, bool);
primitive_cir_type!(IsizeType, isize);
primitive_cir_type!(BigIntType, bigint);
primitive_cir_type!(NilType, nil);
primitive_cir_type!(ConsType, cons);
primitive_cir_type!(MapType, map);
primitive_cir_type!(BitsType, bits);
primitive_cir_type!(BinaryType, binary);
primitive_cir_type!(PidType, pid);
primitive_cir_type!(PortType, port);
primitive_cir_type!(ReferenceType, reference);
primitive_cir_type!(ExceptionType, exception);
primitive_cir_type!(TraceType, trace);
primitive_cir_type!(RecvContextType, recv_context);
primitive_cir_type!(BinaryBuilderType, binary_builder);
generic_cir_type!(BoxType, box);
generic_type!(PtrType, ptr);
container_cir_type!(FunType, fun);

impl AtomLike for AtomType {}
impl AtomLike for BoolType {}

//----------------------------
// Operations
//----------------------------

/// Represents a type casting/conversion operation
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct CastOp(OperationBase);
impl Operation for CastOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_cast<V: Value, T: Type>(&self, loc: Location, value: V, ty: T) -> CastOp {
        extern "C" {
            fn mlirCirCastOp(
                builder: OpBuilderBase,
                location: Location,
                value: ValueBase,
                to_type: TypeBase,
            ) -> CastOp;
        }

        unsafe { mlirCirCastOp(self.base().into(), loc, value.base(), ty.base()) }
    }
}

/// Represents a constant term value
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ConstantOp(OperationBase);
impl Operation for ConstantOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_constant<T: Type, A: Attribute>(
        &self,
        loc: Location,
        ty: T,
        value: A,
    ) -> ConstantOp {
        extern "C" {
            fn mlirCirConstantOp(
                builder: OpBuilderBase,
                loc: Location,
                value: AttributeBase,
                ty: TypeBase,
            ) -> ConstantOp;
        }

        unsafe { mlirCirConstantOp(self.base().into(), loc, value.base(), ty.base()) }
    }
}

/// Represents a constant null value
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ConstantNullOp(OperationBase);
impl Operation for ConstantNullOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_null<T: Type>(&self, loc: Location, ty: T) -> ConstantNullOp {
        extern "C" {
            fn mlirCirConstantNullOp(
                builder: OpBuilderBase,
                loc: Location,
                ty: TypeBase,
            ) -> ConstantNullOp;
        }

        unsafe { mlirCirConstantNullOp(self.base().into(), loc, ty.base()) }
    }
}

/// Represents a null-checking op
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct IsNullOp(OperationBase);
impl Operation for IsNullOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_is_null<V: Value>(&self, loc: Location, value: V) -> IsNullOp {
        extern "C" {
            fn mlirCirIsNullOp(
                builder: OpBuilderBase,
                location: Location,
                value: ValueBase,
            ) -> IsNullOp;
        }

        unsafe { mlirCirIsNullOp(self.base().into(), loc, value.base()) }
    }
}

/// Represents a raw integer truncation
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct TruncOp(OperationBase);
impl Operation for TruncOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_trunc<V: Value>(&self, loc: Location, value: V, ty: TypeBase) -> TruncOp {
        extern "C" {
            fn mlirCirTruncOp(
                builder: OpBuilderBase,
                location: Location,
                value: ValueBase,
                ty: TypeBase,
            ) -> TruncOp;
        }

        unsafe { mlirCirTruncOp(self.base().into(), loc, value.base(), ty) }
    }
}

/// Represents zero-extension of a raw integer
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ZExtOp(OperationBase);
impl Operation for ZExtOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_zext<V: Value>(&self, loc: Location, value: V, ty: TypeBase) -> ZExtOp {
        extern "C" {
            fn mlirCirZExtOp(
                builder: OpBuilderBase,
                location: Location,
                value: ValueBase,
                ty: TypeBase,
            ) -> ZExtOp;
        }

        unsafe { mlirCirZExtOp(self.base().into(), loc, value.base(), ty) }
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ICmpPredicate {
    Eq = 0,
    Neq,
    Gt,
    Gte,
    Lt,
    Lte,
}

#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ICmpOp(OperationBase);
impl Operation for ICmpOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_icmp<L: Value, R: Value>(
        &self,
        loc: Location,
        predicate: ICmpPredicate,
        lhs: L,
        rhs: R,
    ) -> ICmpOp {
        extern "C" {
            fn mlirCirICmpOp(
                builder: OpBuilderBase,
                loc: Location,
                predicate: ICmpPredicate,
                lhs: ValueBase,
                rhs: ValueBase,
            ) -> ICmpOp;
        }

        unsafe { mlirCirICmpOp(self.base().into(), loc, predicate, lhs.base(), rhs.base()) }
    }
}

/// Represents a logical boolean op, e.g. and/andalso, or/orelse, xor, not
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct LogicalOp(OperationBase);
impl Operation for LogicalOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_and<L: Value, R: Value>(&self, loc: Location, lhs: L, rhs: R) -> LogicalOp {
        extern "C" {
            fn mlirCirAndOp(
                builder: OpBuilderBase,
                loc: Location,
                lhs: ValueBase,
                rhs: ValueBase,
            ) -> LogicalOp;
        }

        unsafe { mlirCirAndOp(self.base().into(), loc, lhs.base(), rhs.base()) }
    }

    #[inline]
    pub fn build_andalso<L: Value, R: Value>(&self, loc: Location, lhs: L, rhs: R) -> LogicalOp {
        extern "C" {
            fn mlirCirAndAlsoOp(
                builder: OpBuilderBase,
                loc: Location,
                lhs: ValueBase,
                rhs: ValueBase,
            ) -> LogicalOp;
        }

        unsafe { mlirCirAndAlsoOp(self.base().into(), loc, lhs.base(), rhs.base()) }
    }

    #[inline]
    pub fn build_or<L: Value, R: Value>(&self, loc: Location, lhs: L, rhs: R) -> LogicalOp {
        extern "C" {
            fn mlirCirOrOp(
                builder: OpBuilderBase,
                loc: Location,
                lhs: ValueBase,
                rhs: ValueBase,
            ) -> LogicalOp;
        }

        unsafe { mlirCirOrOp(self.base().into(), loc, lhs.base(), rhs.base()) }
    }

    #[inline]
    pub fn build_orelse<L: Value, R: Value>(&self, loc: Location, lhs: L, rhs: R) -> LogicalOp {
        extern "C" {
            fn mlirCirOrElseOp(
                builder: OpBuilderBase,
                loc: Location,
                lhs: ValueBase,
                rhs: ValueBase,
            ) -> LogicalOp;
        }

        unsafe { mlirCirOrElseOp(self.base().into(), loc, lhs.base(), rhs.base()) }
    }

    #[inline]
    pub fn build_xor<L: Value, R: Value>(&self, loc: Location, lhs: L, rhs: R) -> LogicalOp {
        extern "C" {
            fn mlirCirXorOp(
                builder: OpBuilderBase,
                loc: Location,
                lhs: ValueBase,
                rhs: ValueBase,
            ) -> LogicalOp;
        }

        unsafe { mlirCirXorOp(self.base().into(), loc, lhs.base(), rhs.base()) }
    }

    #[inline]
    pub fn build_not<V: Value>(&self, loc: Location, value: V) -> LogicalOp {
        extern "C" {
            fn mlirCirNotOp(builder: OpBuilderBase, loc: Location, value: ValueBase) -> LogicalOp;
        }

        unsafe { mlirCirNotOp(self.base().into(), loc, value.base()) }
    }
}

/// Represents extraction of the primitive type code from a term
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct TypeOfOp(OperationBase);
impl Operation for TypeOfOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_typeof<V: Value>(&self, loc: Location, value: V) -> TypeOfOp {
        extern "C" {
            fn mlirCirTypeOfOp(builder: OpBuilderBase, loc: Location, value: ValueBase)
                -> TypeOfOp;
        }

        unsafe { mlirCirTypeOfOp(self.base().into(), loc, value.base()) }
    }
}

/// Represents a predicate that answers whether a value is of a given type
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct IsTypeOp(OperationBase);
impl Operation for IsTypeOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_is_type<V: Value, T: Type>(&self, loc: Location, value: V, ty: T) -> IsTypeOp {
        extern "C" {
            fn mlirCirIsTypeOp(
                builder: OpBuilderBase,
                loc: Location,
                value: ValueBase,
                ty: TypeBase,
            ) -> IsTypeOp;
        }

        unsafe { mlirCirIsTypeOp(self.base().into(), loc, value.base(), ty.base()) }
    }
}

/// Represents a predicate that answers whether a value is a list
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct IsListOp(OperationBase);
impl Operation for IsListOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_is_list<V: Value>(&self, loc: Location, value: V) -> IsListOp {
        extern "C" {
            fn mlirCirIsListOp(builder: OpBuilderBase, loc: Location, value: ValueBase)
                -> IsListOp;
        }

        unsafe { mlirCirIsListOp(self.base().into(), loc, value.base()) }
    }
}

/// Represents a predicate that answers whether a value is a number
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct IsNumberOp(OperationBase);
impl Operation for IsNumberOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_is_number<V: Value>(&self, loc: Location, value: V) -> IsNumberOp {
        extern "C" {
            fn mlirCirIsNumberOp(
                builder: OpBuilderBase,
                loc: Location,
                value: ValueBase,
            ) -> IsNumberOp;
        }

        unsafe { mlirCirIsNumberOp(self.base().into(), loc, value.base()) }
    }
}

/// Represents a predicate that answers whether a value is a float
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct IsFloatOp(OperationBase);
impl Operation for IsFloatOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_is_float<V: Value>(&self, loc: Location, value: V) -> IsFloatOp {
        extern "C" {
            fn mlirCirIsFloatOp(
                builder: OpBuilderBase,
                loc: Location,
                value: ValueBase,
            ) -> IsFloatOp;
        }

        unsafe { mlirCirIsFloatOp(self.base().into(), loc, value.base()) }
    }
}

/// Represents a predicate that answers whether a value is an integer
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct IsIntegerOp(OperationBase);
impl Operation for IsIntegerOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_is_integer<V: Value>(&self, loc: Location, value: V) -> IsIntegerOp {
        extern "C" {
            fn mlirCirIsIntegerOp(
                builder: OpBuilderBase,
                loc: Location,
                value: ValueBase,
            ) -> IsIntegerOp;
        }

        unsafe { mlirCirIsIntegerOp(self.base().into(), loc, value.base()) }
    }
}

/// Represents a predicate that answers whether a value is a fixed-width integer
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct IsIsizeOp(OperationBase);
impl Operation for IsIsizeOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_is_isize<V: Value>(&self, loc: Location, value: V) -> IsIsizeOp {
        extern "C" {
            fn mlirCirIsIsizeOp(
                builder: OpBuilderBase,
                loc: Location,
                value: ValueBase,
            ) -> IsIsizeOp;
        }

        unsafe { mlirCirIsIsizeOp(self.base().into(), loc, value.base()) }
    }
}

/// Represents a predicate that answers whether a value is a big integer
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct IsBigIntOp(OperationBase);
impl Operation for IsBigIntOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_is_bigint<V: Value>(&self, loc: Location, value: V) -> IsBigIntOp {
        extern "C" {
            fn mlirCirIsBigIntOp(
                builder: OpBuilderBase,
                loc: Location,
                value: ValueBase,
            ) -> IsBigIntOp;
        }

        unsafe { mlirCirIsBigIntOp(self.base().into(), loc, value.base()) }
    }
}

/// Represents a predicate that answers whether a value is an atom
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct IsAtomOp(OperationBase);
impl Operation for IsAtomOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_is_atom<V: Value>(&self, loc: Location, value: V) -> IsAtomOp {
        extern "C" {
            fn mlirCirIsAtomOp(builder: OpBuilderBase, loc: Location, value: ValueBase)
                -> IsAtomOp;
        }

        unsafe { mlirCirIsAtomOp(self.base().into(), loc, value.base()) }
    }
}

/// Represents a predicate that answers whether a value is a boolean
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct IsBoolOp(OperationBase);
impl Operation for IsBoolOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_is_bool<V: Value>(&self, loc: Location, value: V) -> IsBoolOp {
        extern "C" {
            fn mlirCirIsBoolOp(builder: OpBuilderBase, loc: Location, value: ValueBase)
                -> IsBoolOp;
        }

        unsafe { mlirCirIsBoolOp(self.base().into(), loc, value.base()) }
    }
}

/// Represents a type check for tagged tuples, i.e. it checks that the given value is a tuple,
/// has at least one element, and that the first element is an atom equal to the one given
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct IsTaggedTupleOp(OperationBase);
impl Operation for IsTaggedTupleOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_is_tagged_tuple<V: Value, A: Into<AtomRef>>(
        &self,
        loc: Location,
        value: V,
        tag: A,
    ) -> IsTaggedTupleOp {
        extern "C" {
            fn mlirCirIsTaggedTupleOp(
                builder: OpBuilderBase,
                loc: Location,
                value: ValueBase,
                atom: AtomRef,
            ) -> IsTaggedTupleOp;
        }

        unsafe { mlirCirIsTaggedTupleOp(self.base().into(), loc, value.base(), tag.into()) }
    }
}

/// Represents the allocation of process heap memory backing a value of a specific type
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct MallocOp(OperationBase);
impl Operation for MallocOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_malloc<T: Type>(&self, loc: Location, alloc_type: T) -> MallocOp {
        extern "C" {
            fn mlirCirMallocOp(
                builder: OpBuilderBase,
                loc: Location,
                alloc_type: TypeBase,
            ) -> MallocOp;
        }

        unsafe { mlirCirMallocOp(self.base().into(), loc, alloc_type.base()) }
    }
}

/// Represents the construction of a fun/closure
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct MakeFunOp(OperationBase);
impl Operation for MakeFunOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_fun(&self, loc: Location, callee: FuncOp, env: &[ValueBase]) -> MakeFunOp {
        extern "C" {
            fn mlirCirMakeFunOp(
                builder: OpBuilderBase,
                loc: Location,
                callee: FuncOp,
                env: *const ValueBase,
                env_len: usize,
            ) -> MakeFunOp;
        }

        unsafe { mlirCirMakeFunOp(self.base().into(), loc, callee, env.as_ptr(), env.len()) }
    }
}

/// Represents the construction of a cons cell
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ConsOp(OperationBase);
impl Operation for ConsOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_cons<H: Value, T: Value>(&self, loc: Location, head: H, tail: T) -> ConsOp {
        extern "C" {
            fn mlirCirConsOp(
                builder: OpBuilderBase,
                loc: Location,
                head: ValueBase,
                tail: ValueBase,
            ) -> ConsOp;
        }

        unsafe { mlirCirConsOp(self.base().into(), loc, head.base(), tail.base()) }
    }
}

/// Represents destructuring of a cons cell's head value
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct HeadOp(OperationBase);
impl Operation for HeadOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_head<V: Value>(&self, loc: Location, value: V) -> HeadOp {
        extern "C" {
            fn mlirCirHeadOp(builder: OpBuilderBase, loc: Location, cons: ValueBase) -> HeadOp;
        }

        unsafe { mlirCirHeadOp(self.base().into(), loc, value.base()) }
    }
}

/// Represents destructuring of a cons cell's tail value
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct TailOp(OperationBase);
impl Operation for TailOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_tail<V: Value>(&self, loc: Location, value: V) -> TailOp {
        extern "C" {
            fn mlirCirTailOp(builder: OpBuilderBase, loc: Location, cons: ValueBase) -> TailOp;
        }

        unsafe { mlirCirTailOp(self.base().into(), loc, value.base()) }
    }
}

/// Represents construction of a tuple of a specific arity
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct TupleOp(OperationBase);
impl Operation for TupleOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_tuple(&self, loc: Location, arity: usize) -> TupleOp {
        extern "C" {
            fn mlirCirTupleOp(builder: OpBuilderBase, loc: Location, arity: usize) -> TupleOp;
        }

        unsafe { mlirCirTupleOp(self.base().into(), loc, arity) }
    }
}

/// Represents setting the element of a tuple to a specific value
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct SetElementOp(OperationBase);
impl Operation for SetElementOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_set_element<T: Value, I: Value, V: Value>(
        &self,
        loc: Location,
        tuple: T,
        index: I,
        value: V,
    ) -> SetElementOp {
        extern "C" {
            fn mlirCirSetElementOp(
                builder: OpBuilderBase,
                loc: Location,
                tuple: ValueBase,
                index: ValueBase,
                value: ValueBase,
            ) -> SetElementOp;
        }

        unsafe {
            mlirCirSetElementOp(
                self.base().into(),
                loc,
                tuple.base(),
                index.base(),
                value.base(),
            )
        }
    }
}

/// Represents getting the element of a tuple at a specific index
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct GetElementOp(OperationBase);
impl Operation for GetElementOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_get_element<V: Value, I: Value>(
        &self,
        loc: Location,
        tuple: V,
        index: I,
    ) -> GetElementOp {
        extern "C" {
            fn mlirCirGetElementOp(
                builder: OpBuilderBase,
                loc: Location,
                tuple: ValueBase,
                index: ValueBase,
            ) -> GetElementOp;
        }

        unsafe { mlirCirGetElementOp(self.base().into(), loc, tuple.base(), index.base()) }
    }
}

/// Represents creating an empty map
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct MapOp(OperationBase);
impl Operation for MapOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_map(&self, loc: Location) -> MapOp {
        extern "C" {
            fn mlirCirMapOp(builder: OpBuilderBase, loc: Location) -> MapOp;
        }

        unsafe { mlirCirMapOp(self.base().into(), loc) }
    }
}

/// Represents fetching a key from a map
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct MapGetOp(OperationBase);
impl Operation for MapGetOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_map_get<M: Value, K: Value>(&self, loc: Location, map: M, key: K) -> MapGetOp {
        extern "C" {
            fn mlirCirMapGetOp(
                builder: OpBuilderBase,
                loc: Location,
                map: ValueBase,
                key: ValueBase,
            ) -> MapGetOp;
        }

        unsafe { mlirCirMapGetOp(self.base().into(), loc, map.base(), key.base()) }
    }
}

/// Represents raising an Erlang exception
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct RaiseOp(OperationBase);
impl Operation for RaiseOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_raise<C: Value, R: Value, T: Value>(
        &self,
        loc: Location,
        class: C,
        reason: R,
        trace: T,
    ) -> RaiseOp {
        extern "C" {
            fn mlirCirRaiseOp(
                builder: OpBuilderBase,
                loc: Location,
                class: ValueBase,
                reason: ValueBase,
                trace: ValueBase,
            ) -> RaiseOp;
        }

        unsafe {
            mlirCirRaiseOp(
                self.base().into(),
                loc,
                class.base(),
                reason.base(),
                trace.base(),
            )
        }
    }
}

/// Represents capturing an erlang stacktrace at the current program counter
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct BuildStacktraceOp(OperationBase);
impl Operation for BuildStacktraceOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_stacktrace(&self, loc: Location) -> BuildStacktraceOp {
        extern "C" {
            fn mlirCirBuildStacktraceOp(builder: OpBuilderBase, loc: Location)
                -> BuildStacktraceOp;
        }

        unsafe { mlirCirBuildStacktraceOp(self.base().into(), loc) }
    }
}

/// Represents accessing the class of an Erlang exception
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ExceptionClassOp(OperationBase);
impl Operation for ExceptionClassOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_exception_class<V: Value>(&self, loc: Location, exception: V) -> ExceptionClassOp {
        extern "C" {
            fn mlirCirExceptionClassOp(
                builder: OpBuilderBase,
                loc: Location,
                exception: ValueBase,
            ) -> ExceptionClassOp;
        }

        unsafe { mlirCirExceptionClassOp(self.base().into(), loc, exception.base()) }
    }
}

/// Represents accessing the reason of an Erlang exception
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ExceptionReasonOp(OperationBase);
impl Operation for ExceptionReasonOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_exception_reason<V: Value>(
        &self,
        loc: Location,
        exception: V,
    ) -> ExceptionReasonOp {
        extern "C" {
            fn mlirCirExceptionReasonOp(
                builder: OpBuilderBase,
                loc: Location,
                exception: ValueBase,
            ) -> ExceptionReasonOp;
        }

        unsafe { mlirCirExceptionReasonOp(self.base().into(), loc, exception.base()) }
    }
}

/// Represents accessing the trace associated to an Erlang exception
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ExceptionTraceOp(OperationBase);
impl Operation for ExceptionTraceOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_exception_trace<V: Value>(&self, loc: Location, exception: V) -> ExceptionTraceOp {
        extern "C" {
            fn mlirCirExceptionTraceOp(
                builder: OpBuilderBase,
                loc: Location,
                exception: ValueBase,
            ) -> ExceptionTraceOp;
        }

        unsafe { mlirCirExceptionTraceOp(self.base().into(), loc, exception.base()) }
    }
}

/// Represents yielding control to the scheduler from the current process
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct YieldOp(OperationBase);
impl Operation for YieldOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_yield(&self, loc: Location) -> YieldOp {
        extern "C" {
            fn mlirCirYieldOp(builder: OpBuilderBase, loc: Location) -> YieldOp;
        }

        unsafe { mlirCirYieldOp(self.base().into(), loc) }
    }
}

/// Represents initializing a receive block
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct RecvStartOp(OperationBase);
impl Operation for RecvStartOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_recv_start<V: Value>(&self, loc: Location, timeout: V) -> RecvStartOp {
        extern "C" {
            fn mlirCirRecvStartOp(
                builder: OpBuilderBase,
                loc: Location,
                timeout: ValueBase,
            ) -> RecvStartOp;
        }

        unsafe { mlirCirRecvStartOp(self.base().into(), loc, timeout.base()) }
    }
}

/// Represents selecting the next available message in the mailbox to peek
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct RecvNextOp(OperationBase);
impl Operation for RecvNextOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_recv_next(&self, loc: Location, recv_context: ValueBase) -> RecvNextOp {
        extern "C" {
            fn mlirCirRecvNextOp(
                builder: OpBuilderBase,
                loc: Location,
                recv_context: ValueBase,
            ) -> RecvNextOp;
        }

        unsafe { mlirCirRecvNextOp(self.base().into(), loc, recv_context.base()) }
    }
}

/// Represents fetching a message from the mailbox pointed to by the current receive context
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct RecvPeekOp(OperationBase);
impl Operation for RecvPeekOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_recv_peek(&self, loc: Location, recv_context: ValueBase) -> RecvPeekOp {
        extern "C" {
            fn mlirCirRecvPeekOp(
                builder: OpBuilderBase,
                loc: Location,
                recv_context: ValueBase,
            ) -> RecvPeekOp;
        }

        unsafe { mlirCirRecvPeekOp(self.base().into(), loc, recv_context.base()) }
    }
}

/// Represents removing a message from the mailbox pointed to by the current receive context
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct RecvPopOp(OperationBase);
impl Operation for RecvPopOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_recv_pop(&self, loc: Location, recv_context: ValueBase) -> RecvPopOp {
        extern "C" {
            fn mlirCirRecvPopOp(
                builder: OpBuilderBase,
                loc: Location,
                recv_context: ValueBase,
            ) -> RecvPopOp;
        }

        unsafe { mlirCirRecvPopOp(self.base().into(), loc, recv_context.base()) }
    }
}

/// Represents exiting a receive block and performing necessary cleanup of the receive context
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct RecvDoneOp(OperationBase);
impl Operation for RecvDoneOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_recv_done(&self, loc: Location, recv_context: ValueBase) -> RecvDoneOp {
        extern "C" {
            fn mlirCirRecvDoneOp(
                builder: OpBuilderBase,
                loc: Location,
                recv_context: ValueBase,
            ) -> RecvDoneOp;
        }

        unsafe { mlirCirRecvDoneOp(self.base().into(), loc, recv_context.base()) }
    }
}

/// Represents the binary builder context used with binary construction primops
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct BinaryBuilder(OpResult);
impl Value for BinaryBuilder {
    #[inline(always)]
    fn base(&self) -> ValueBase {
        self.0.base()
    }
}

/// Represents initializing construction of a new binary value
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct BinaryInitOp(OperationBase);
impl Operation for BinaryInitOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_binary_init(&self, loc: Location) -> BinaryInitOp {
        extern "C" {
            fn mlirCirBinaryInitOp(builder: OpBuilderBase, loc: Location) -> BinaryInitOp;
        }

        unsafe { mlirCirBinaryInitOp(self.base().into(), loc) }
    }
}

/// Represents converting a binary builder into a fully constructed term value
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct BinaryFinishOp(OperationBase);
impl Operation for BinaryFinishOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_binary_finish(&self, loc: Location, bin_builder: ValueBase) -> BinaryFinishOp {
        extern "C" {
            fn mlirCirBinaryFinishOp(
                builder: OpBuilderBase,
                loc: Location,
                bin_builder: ValueBase,
            ) -> BinaryFinishOp;
        }

        unsafe { mlirCirBinaryFinishOp(self.base().into(), loc, bin_builder) }
    }
}

/// Represents pushing an integer value on to a binary builder
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct BinaryPushIntegerOp(OperationBase);
impl Operation for BinaryPushIntegerOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    /// Pushes the given value as an integer, using the provided details to specify the encoding
    #[inline]
    pub fn build_binary_push_integer<V: Value, S: Value>(
        &self,
        loc: Location,
        bin_builder: ValueBase,
        value: V,
        size: S,
        is_signed: bool,
        endianness: Endianness,
        unit: u32,
    ) -> BinaryPushIntegerOp {
        extern "C" {
            fn mlirCirBinaryPushIntegerOp(
                builder: OpBuilderBase,
                loc: Location,
                bin_builder: ValueBase,
                value: ValueBase,
                size: ValueBase,
                is_signed: bool,
                endianness: Endianness,
                unit: u32,
            ) -> BinaryPushIntegerOp;
        }

        unsafe {
            mlirCirBinaryPushIntegerOp(
                self.base().into(),
                loc,
                bin_builder.base(),
                value.base(),
                size.base(),
                is_signed,
                endianness,
                unit,
            )
        }
    }
}

/// Represents pushing a floating-point value on to a binary builder
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct BinaryPushFloatOp(OperationBase);
impl Operation for BinaryPushFloatOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_binary_push_float<V: Value, S: Value>(
        &self,
        loc: Location,
        bin_builder: ValueBase,
        value: V,
        size: S,
        endianness: Endianness,
        unit: u32,
    ) -> BinaryPushFloatOp {
        extern "C" {
            fn mlirCirBinaryPushFloatOp(
                builder: OpBuilderBase,
                loc: Location,
                bin_builder: ValueBase,
                value: ValueBase,
                size: ValueBase,
                endianness: Endianness,
                unit: u32,
            ) -> BinaryPushFloatOp;
        }

        unsafe {
            mlirCirBinaryPushFloatOp(
                self.base().into(),
                loc,
                bin_builder.base(),
                value.base(),
                size.base(),
                endianness,
                unit,
            )
        }
    }
}

/// Represents pushing a utf-8 codepoint on to a binary builder
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct BinaryPushUtf8Op(OperationBase);
impl Operation for BinaryPushUtf8Op {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_binary_push_utf8<V: Value>(
        &self,
        loc: Location,
        bin_builder: ValueBase,
        value: V,
    ) -> BinaryPushUtf8Op {
        extern "C" {
            fn mlirCirBinaryPushUtf8Op(
                builder: OpBuilderBase,
                loc: Location,
                bin_builder: ValueBase,
                value: ValueBase,
            ) -> BinaryPushUtf8Op;
        }

        unsafe {
            mlirCirBinaryPushUtf8Op(self.base().into(), loc, bin_builder.base(), value.base())
        }
    }
}

/// Represents pushing a utf-16 codepoint on to a binary builder
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct BinaryPushUtf16Op(OperationBase);
impl Operation for BinaryPushUtf16Op {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_binary_push_utf16<V: Value>(
        &self,
        loc: Location,
        bin_builder: ValueBase,
        value: V,
        endianness: Endianness,
    ) -> BinaryPushUtf16Op {
        extern "C" {
            fn mlirCirBinaryPushUtf16Op(
                builder: OpBuilderBase,
                loc: Location,
                bin_builder: ValueBase,
                value: ValueBase,
                endianness: Endianness,
            ) -> BinaryPushUtf16Op;
        }

        unsafe {
            mlirCirBinaryPushUtf16Op(
                self.base().into(),
                loc,
                bin_builder.base(),
                value.base(),
                endianness,
            )
        }
    }
}

/// Represents pushing a utf-32 codepoint on to a binary builder
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct BinaryPushUtf32Op(OperationBase);
impl Operation for BinaryPushUtf32Op {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_binary_push_utf32<V: Value>(
        &self,
        loc: Location,
        bin_builder: ValueBase,
        value: V,
        endianness: Endianness,
    ) -> BinaryPushUtf16Op {
        extern "C" {
            fn mlirCirBinaryPushUtf32Op(
                builder: OpBuilderBase,
                loc: Location,
                bin_builder: ValueBase,
                value: ValueBase,
                endianness: Endianness,
            ) -> BinaryPushUtf16Op;
        }

        unsafe {
            mlirCirBinaryPushUtf32Op(
                self.base().into(),
                loc,
                bin_builder.base(),
                value.base(),
                endianness,
            )
        }
    }
}

/// Represents pushing a binary/bitstring value on to a binary builder
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct BinaryPushBitsOp(OperationBase);
impl Operation for BinaryPushBitsOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_binary_push_bits_all<V: Value>(
        &self,
        loc: Location,
        bin_builder: ValueBase,
        value: V,
    ) -> BinaryPushBitsOp {
        extern "C" {
            fn mlirCirBinaryPushBitsAllOp(
                builder: OpBuilderBase,
                loc: Location,
                bin_builder: ValueBase,
                value: ValueBase,
            ) -> BinaryPushBitsOp;
        }

        unsafe {
            mlirCirBinaryPushBitsAllOp(self.base().into(), loc, bin_builder.base(), value.base())
        }
    }

    #[inline]
    pub fn build_binary_push_bits<V: Value, S: Value>(
        &self,
        loc: Location,
        bin_builder: ValueBase,
        value: V,
        size: S,
        unit: u32,
    ) -> BinaryPushBitsOp {
        extern "C" {
            fn mlirCirBinaryPushBitsOp(
                builder: OpBuilderBase,
                loc: Location,
                bin_builder: ValueBase,
                value: ValueBase,
                size: ValueBase,
                unit: u32,
            ) -> BinaryPushBitsOp;
        }

        unsafe {
            mlirCirBinaryPushBitsOp(
                self.base().into(),
                loc,
                bin_builder.base(),
                value.base(),
                size.base(),
                unit,
            )
        }
    }
}

/// Represents pushing a term of any type on to a binary builder, optionally with a specified size
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct BinaryPushAnyOp(OperationBase);
impl Operation for BinaryPushAnyOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_binary_push_any<V: Value>(
        &self,
        loc: Location,
        bin_builder: ValueBase,
        value: V,
    ) -> BinaryPushAnyOp {
        extern "C" {
            fn mlirCirBinaryPushAnyOp(
                builder: OpBuilderBase,
                loc: Location,
                bin_builder: ValueBase,
                value: ValueBase,
            ) -> BinaryPushAnyOp;
        }

        unsafe { mlirCirBinaryPushAnyOp(self.base().into(), loc, bin_builder.base(), value.base()) }
    }

    #[inline]
    pub fn build_binary_push_any_sized<V: Value, S: Value>(
        &self,
        loc: Location,
        bin_builder: ValueBase,
        value: V,
        size: S,
    ) -> BinaryPushAnyOp {
        extern "C" {
            fn mlirCirBinaryPushAnySizedOp(
                builder: OpBuilderBase,
                loc: Location,
                bin_builder: ValueBase,
                value: ValueBase,
                size: ValueBase,
            ) -> BinaryPushAnyOp;
        }

        unsafe {
            mlirCirBinaryPushAnySizedOp(
                self.base().into(),
                loc,
                bin_builder.base(),
                value.base(),
                size.base(),
            )
        }
    }
}

/// Represents the dispatch table associated with a module
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct DispatchTableOp(OperationBase);
impl DispatchTableOp {
    pub fn append(
        &self,
        loc: Location,
        function: StringAttr,
        arity: IntegerAttr,
        symbol: FlatSymbolRefAttr,
    ) {
        extern "C" {
            fn mlirCirDispatchTableAppendEntry(op: OperationBase, entry: OperationBase);
        }

        let mut state = OperationState::get("cir.dispatch_entry", loc);
        let context = symbol.context();
        let function = NamedAttribute::get(StringAttr::get(context, "function"), function);
        let arity = NamedAttribute::get(StringAttr::get(context, "arity"), arity);
        let symbol = NamedAttribute::get(StringAttr::get(context, "symbol"), symbol);
        state.add_attributes(&[function, arity, symbol]);
        let entry = state.create().release();

        unsafe {
            mlirCirDispatchTableAppendEntry(self.0, entry);
        }
    }
}
impl Operation for DispatchTableOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_dispatch_table<S: Into<StringRef>>(
        &self,
        loc: Location,
        module: S,
    ) -> DispatchTableOp {
        extern "C" {
            fn mlirCirDispatchTableOp(
                builder: OpBuilderBase,
                loc: Location,
                module: StringRef,
            ) -> DispatchTableOp;
        }

        unsafe { mlirCirDispatchTableOp(self.base().into(), loc, module.into()) }
    }

    pub fn build_dispatch_entry<S: Into<StringRef>>(
        &self,
        loc: Location,
        table: DispatchTableOp,
        function: S,
        arity: u8,
        symbol: FlatSymbolRefAttr,
    ) {
        let function = self.builder.get_string_attr(function);
        let arity = self.builder.get_i8_attr(arity as i8);
        table.append(loc, function, arity, symbol);
    }
}

/// Represents a direct call to a function symbol, potentially outside the current module
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct CallOp(OperationBase);
impl CallOp {
    pub fn callee(&self) -> attributes::FlatSymbolRefAttr {
        let attr = self.get_attribute_by_name("callee").unwrap();
        attr.try_into().unwrap()
    }

    pub fn callee_type(&self) -> types::FunctionType {
        let operands = self.operands().map(|v| v.get_type()).collect::<Vec<_>>();
        let results = self.results().map(|r| r.get_type()).collect::<Vec<_>>();
        types::FunctionType::get(self.context(), operands.as_slice(), results.as_slice())
    }
}
impl Operation for CallOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}

impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_call(&self, loc: Location, func: FuncOp, args: &[ValueBase]) -> CallOp {
        extern "C" {
            fn mlirFuncCallByOp(
                builder: OpBuilderBase,
                loc: Location,
                callee: FuncOp,
                num_args: usize,
                args: *const ValueBase,
            ) -> CallOp;
        }
        unsafe { mlirFuncCallByOp(self.base().into(), loc, func, args.len(), args.as_ptr()) }
    }

    #[inline]
    pub fn build_call_by_name(
        &self,
        loc: Location,
        callee: attributes::FlatSymbolRefAttr,
        results: &[TypeBase],
        args: &[ValueBase],
    ) -> CallOp {
        extern "C" {
            fn mlirFuncCallBySymbol(
                builder: OpBuilderBase,
                loc: Location,
                callee: attributes::FlatSymbolRefAttr,
                num_results: usize,
                results: *const TypeBase,
                num_args: usize,
                args: *const ValueBase,
            ) -> CallOp;
        }

        unsafe {
            mlirFuncCallBySymbol(
                self.base().into(),
                loc,
                callee,
                results.len(),
                results.as_ptr(),
                args.len(),
                args.as_ptr(),
            )
        }
    }
}

/// Represents an indirect call to a function value
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct CallIndirectOp(OperationBase);
impl CallIndirectOp {
    pub fn callee(&self) -> ValueBase {
        self.0.get_operand(0)
    }

    pub fn callee_type(&self) -> types::FunctionType {
        self.callee().get_type().try_into().unwrap()
    }
}
impl Operation for CallIndirectOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}

impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_call_indirect(
        &self,
        loc: Location,
        callee: ValueBase,
        args: &[ValueBase],
    ) -> CallIndirectOp {
        extern "C" {
            fn mlirFuncCallIndirect(
                builder: OpBuilderBase,
                loc: Location,
                callee: ValueBase,
                num_args: usize,
                args: *const ValueBase,
            ) -> CallIndirectOp;
        }

        unsafe { mlirFuncCallIndirect(self.base().into(), loc, callee, args.len(), args.as_ptr()) }
    }
}

/// Represents a return from the enclosing function
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ReturnOp(OperationBase);
impl Operation for ReturnOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}

impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_return(&self, loc: Location, results: &[ValueBase]) -> ReturnOp {
        extern "C" {
            fn mlirFuncReturn(
                builder: OpBuilderBase,
                loc: Location,
                num_results: usize,
                results: *const ValueBase,
            ) -> ReturnOp;
        }

        unsafe { mlirFuncReturn(self.base().into(), loc, results.len(), results.as_ptr()) }
    }
}

/// Represents an unconditional jump
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct BranchOp(OperationBase);
impl BranchOp {
    pub fn dest(&self) -> Block {
        self.0.get_successor(0)
    }
}
impl Operation for BranchOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}

impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_branch(&self, loc: Location, dest: Block, args: &[ValueBase]) -> BranchOp {
        extern "C" {
            fn mlirControlFlowBranch(
                builder: OpBuilderBase,
                loc: Location,
                dest: Block,
                num_args: usize,
                args: *const ValueBase,
            ) -> BranchOp;
        }

        unsafe { mlirControlFlowBranch(self.base().into(), loc, dest, args.len(), args.as_ptr()) }
    }
}

/// Represents a conditional jump
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct CondBranchOp(OperationBase);
impl CondBranchOp {
    pub fn true_dest(&self) -> Block {
        self.0.get_successor(0)
    }

    pub fn false_dest(&self) -> Block {
        self.0.get_successor(1)
    }
}
impl Operation for CondBranchOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}

impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_cond_branch<V: Value>(
        &self,
        loc: Location,
        cond: V,
        true_dest: Block,
        true_args: &[ValueBase],
        false_dest: Block,
        false_args: &[ValueBase],
    ) -> CondBranchOp {
        extern "C" {
            fn mlirControlFlowCondBranch(
                builder: OpBuilderBase,
                loc: Location,
                cond: ValueBase,
                true_dest: Block,
                num_true_args: usize,
                true_args: *const ValueBase,
                false_dest: Block,
                num_false_args: usize,
                false_args: *const ValueBase,
            ) -> CondBranchOp;
        }

        unsafe {
            mlirControlFlowCondBranch(
                self.base().into(),
                loc,
                cond.base(),
                true_dest,
                true_args.len(),
                true_args.as_ptr(),
                false_dest,
                false_args.len(),
                false_args.as_ptr(),
            )
        }
    }
}

#[repr(C)]
struct SwitchArm {
    value: u32,
    dest: Block,
    operands: *const ValueBase,
    num_operands: usize,
}

/// Used to construct a SwitchOp safely by ensuring all invariants are upheld
pub struct SwitchBuilder<'a> {
    builder: OpBuilderBase,
    loc: Location,
    input: ValueBase,
    arms: Vec<SwitchArm>,
    operands: Vec<ValueBase>,
    default: Block,
    default_operands: Vec<ValueBase>,
    _marker: core::marker::PhantomData<&'a OpBuilderBase>,
}
impl<'a> SwitchBuilder<'a> {
    /// Starts construction of a new SwitchOp using the provided builder, location, and input value
    pub fn new<B: OpBuilder, V: Value>(
        builder: &'a CirBuilder<'_, B>,
        loc: Location,
        input: V,
    ) -> Self {
        Self {
            builder: builder.base().into(),
            loc,
            input: input.base(),
            arms: vec![],
            operands: vec![],
            default: Block::default(),
            default_operands: vec![],
            _marker: core::marker::PhantomData,
        }
    }

    /// Extends the switch with a new case
    ///
    /// NOTE: The case value must be unique in the switch, if it is not, this function will panic
    pub fn with_case(&mut self, case: u32, dest: Block, operands: &[ValueBase]) -> &mut Self {
        assert!(
            !self.arms.iter().any(|arm| arm.value == case),
            "attempted to define duplicate cases in switch!"
        );

        let start = self.operands.len();
        self.operands.extend_from_slice(operands);
        let num_operands = operands.len();
        let operands = unsafe { self.operands.as_ptr().add(start) };
        self.arms.push(SwitchArm {
            value: case,
            dest,
            operands,
            num_operands,
        });

        self
    }

    /// Extends the switch with a default case
    ///
    /// NOTE: A default case is required, and must be provided prior to `build`, or creation will fail
    pub fn with_default(&mut self, dest: Block, operands: &[ValueBase]) -> &mut Self {
        if self.default.is_null() {
            self.default = dest;
            self.default_operands.extend_from_slice(operands);
        } else {
            self.default = dest;
            self.default_operands.clear();
            self.default_operands.extend_from_slice(operands);
        }
        self
    }

    /// Builds the SwitchOp using the data provided.
    ///
    /// NOTE: If no default case was set, this will panic
    pub fn build(self) -> SwitchOp {
        assert!(!self.default.is_null());
        extern "C" {
            fn mlirControlFlowSwitchOp(
                builder: OpBuilderBase,
                loc: Location,
                value: ValueBase,
                default_dest: Block,
                default_args: *const ValueBase,
                num_default_args: usize,
                arms: *const SwitchArm,
                num_arms: usize,
            ) -> SwitchOp;
        }

        let op = unsafe {
            mlirControlFlowSwitchOp(
                self.builder,
                self.loc,
                self.input.base(),
                self.default,
                self.default_operands.as_ptr(),
                self.default_operands.len(),
                self.arms.as_ptr(),
                self.arms.len(),
            )
        };
        assert!(!op.0.is_null());
        op
    }
}

/// Represents a switch statement
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct SwitchOp(OperationBase);
impl Operation for SwitchOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    /// Obtains a SwitchBuilder which can be used to construct a SwitchOp with this builder
    #[inline]
    pub fn build_switch<V: Value>(&self, loc: Location, value: V) -> SwitchBuilder<'_> {
        SwitchBuilder::new(self, loc, value)
    }
}

/// Represents an if/then/else statement
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct IfOp(OperationBase);
impl IfOp {
    /// Returns the block visited when the input condition is true
    pub fn then_block(&self) -> Block {
        self.get_region(0).entry().unwrap()
    }

    /// If this IfOp was constructed with an else block (the default), this function
    /// returns the block visited when the input condition is false
    pub fn else_block(&self) -> Option<Block> {
        self.get_region(1).entry()
    }
}
impl Operation for IfOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_if<V: Value>(&self, loc: Location, cond: V, results: &[TypeBase]) -> IfOp {
        self.build_if_else(loc, cond, results, /*with_else=*/ false)
    }

    #[inline]
    pub fn build_if_else<V: Value>(
        &self,
        loc: Location,
        cond: V,
        results: &[TypeBase],
        with_else: bool,
    ) -> IfOp {
        extern "C" {
            fn mlirScfIfOp(
                builder: OpBuilderBase,
                loc: Location,
                results: *const TypeBase,
                num_results: usize,
                cond: ValueBase,
                with_else: bool,
            ) -> IfOp;
        }

        unsafe {
            mlirScfIfOp(
                self.base().into(),
                loc,
                results.as_ptr(),
                results.len(),
                cond.base(),
                with_else,
            )
        }
    }
}

/// Represents a for loop
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ForOp(OperationBase);
impl Operation for ForOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_for<L: Value, U: Value, S: Value>(
        &self,
        loc: Location,
        lower_bound: L,
        upper_bound: U,
        step: S,
        init: &[ValueBase],
    ) -> ForOp {
        extern "C" {
            fn mlirScfForOp(
                builder: OpBuilderBase,
                loc: Location,
                lower_bound: ValueBase,
                upper_bound: ValueBase,
                step: ValueBase,
                init: *const ValueBase,
                num_init: usize,
            ) -> ForOp;
        }

        unsafe {
            mlirScfForOp(
                self.base().into(),
                loc,
                lower_bound.base(),
                upper_bound.base(),
                step.base(),
                init.as_ptr(),
                init.len(),
            )
        }
    }
}

/// This op is used to support multiple blocks in a region nested within IfOp/ForOp
///
/// Both of the ops mentioned only permit a single block in their regions, but since more complex control flow
/// is commonly needed in the body of an if/for expression, it is necessary to use this op which can contain
/// arbitrarily many blocks in its one region. It accepts no operands, but operations within are able to access
/// all SSA values that dominate it directly.
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ExecuteRegionOp(OperationBase);
impl Operation for ExecuteRegionOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_execute_region(&self, loc: Location, results: &[TypeBase]) -> ExecuteRegionOp {
        extern "C" {
            fn mlirScfExecuteRegionOp(
                builder: OpBuilderBase,
                loc: Location,
                results: *const TypeBase,
                num_results: usize,
            ) -> ExecuteRegionOp;
        }

        unsafe { mlirScfExecuteRegionOp(self.base().into(), loc, results.as_ptr(), results.len()) }
    }
}

/// Represents return from within a region of an IfOp/ForOp to the containing region
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ScfYieldOp(OperationBase);
impl Operation for ScfYieldOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl<'a, B: OpBuilder> CirBuilder<'a, B> {
    #[inline]
    pub fn build_scf_yield(&self, loc: Location, results: &[ValueBase]) -> ScfYieldOp {
        extern "C" {
            fn mlirScfYieldOp(
                builder: OpBuilderBase,
                loc: Location,
                results: *const ValueBase,
                num_results: usize,
            ) -> ScfYieldOp;
        }

        unsafe { mlirScfYieldOp(self.base().into(), loc, results.as_ptr(), results.len()) }
    }
}

//----------------------------
// Support
//----------------------------

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct AtomRef {
    symbol: usize,
    data: *const u8,
    len: usize,
}
impl ::std::fmt::Display for AtomRef {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        let s = std::str::from_utf8(unsafe { std::slice::from_raw_parts(self.data, self.len) })
            .unwrap();
        write!(f, "{}", s)
    }
}
impl From<Symbol> for AtomRef {
    fn from(atom: Symbol) -> Self {
        let symbol = atom.as_usize();
        let s = atom.as_str().get();
        let bytes = s.as_bytes();
        Self {
            symbol,
            data: bytes.as_ptr(),
            len: bytes.len(),
        }
    }
}
