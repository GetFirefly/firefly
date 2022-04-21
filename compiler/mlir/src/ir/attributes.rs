use std::ffi::c_void;
use std::fmt::{self, Display};

use paste::paste;

use crate::support::{self, MlirStringCallback, StringRef};
use crate::Context;

use super::*;

extern "C" {
    type MlirAttribute;
}

/// This trait represents the concrete implementation of an MLIR attribute
pub trait Attribute {
    /// Returns the MLIR context this attribute was created in
    fn context(&self) -> Context {
        unsafe { mlir_attribute_get_context(self.base()) }
    }
    /// Returns the type assigned to this attribute
    fn get_type(&self) -> TypeBase {
        unsafe { mlir_attribute_get_type(self.base()) }
    }
    /// Dumps this attribute to stderr using its textual representation
    fn dump(&self) {
        unsafe { mlir_attribute_dump(self.base()) }
    }
    /// Returns the underlying AttributeBase value of this attribute
    fn base(&self) -> AttributeBase;
}

/// Represents an key/value pair where the key is a name,
/// and the value is any MLIR attribute.
///
/// These are used to associate known attributes with operations,
/// such as the visibility and symbol of a function operation.
#[repr(C)]
pub struct NamedAttribute {
    name: StringAttr,
    attr: AttributeBase,
}
impl Attribute for NamedAttribute {
    fn base(&self) -> AttributeBase {
        self.attr
    }
}
impl NamedAttribute {
    /// Associates an attribute with the given name.
    ///
    /// NOTE: The NamedAttribute does not take ownership of
    /// either the name, or the attribute value.
    #[inline]
    pub fn get<A: Attribute>(name: StringAttr, value: A) -> Self {
        unsafe { mlir_named_attribute_get(name, value.base()) }
    }

    #[inline(always)]
    pub fn is_null(&self) -> bool {
        self.attr.is_null()
    }

    #[inline(always)]
    pub fn name(&self) -> StringRef {
        self.name.value()
    }

    #[inline(always)]
    pub fn attr(&self) -> AttributeBase {
        self.attr
    }
}

extern "C" {
    #[link_name = "mlirNamedAttributeGet"]
    fn mlir_named_attribute_get(name: StringAttr, attr: AttributeBase) -> NamedAttribute;
}

/// Represents a reference to an MLIR attribute
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct AttributeBase(*mut MlirAttribute);
impl Attribute for AttributeBase {
    #[inline(always)]
    fn base(&self) -> AttributeBase {
        *self
    }
}
impl AttributeBase {
    /// Checks whether a type is null
    #[inline]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Returns true if this type is an instance of the concrete type `T`
    #[inline(always)]
    pub fn isa<T>(self) -> bool
    where
        T: TryFrom<Self>,
    {
        T::try_from(self).is_ok()
    }

    /// Tries to convert this type to an instance of the concrete type `T`
    #[inline(always)]
    pub fn dyn_cast<T>(self) -> Result<T, InvalidTypeCastError>
    where
        T: TryFrom<Self, Error = InvalidTypeCastError>,
    {
        T::try_from(self)
    }
}
impl Default for AttributeBase {
    fn default() -> Self {
        Self(unsafe { std::mem::transmute::<*mut (), *mut MlirAttribute>(::core::ptr::null_mut()) })
    }
}
impl Display for AttributeBase {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe {
            mlir_attribute_print(
                *self,
                support::write_to_formatter,
                f as *mut _ as *mut c_void,
            );
        }
        Ok(())
    }
}
impl fmt::Pointer for AttributeBase {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", self.0)
    }
}
impl Eq for AttributeBase {}
impl PartialEq for AttributeBase {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlir_attribute_equal(*self, *other) }
    }
}

extern "C" {
    #[link_name = "mlirAttributeGetContext"]
    fn mlir_attribute_get_context(attr: AttributeBase) -> Context;
    #[link_name = "mlirAttributeGetType"]
    fn mlir_attribute_get_type(attr: AttributeBase) -> TypeBase;
    #[link_name = "mlirAttributeEqual"]
    fn mlir_attribute_equal(a: AttributeBase, b: AttributeBase) -> bool;
    #[link_name = "mlirAttributePrint"]
    fn mlir_attribute_print(
        attr: AttributeBase,
        callback: MlirStringCallback,
        userdata: *const c_void,
    );
    #[link_name = "mlirAttributeDump"]
    fn mlir_attribute_dump(attr: AttributeBase);
}

macro_rules! primitive_builtin_attr {
    ($name:ident, $mnemonic:ident) => {
        paste! {
            primitive_builtin_attr_impl!($name, $mnemonic, [<$name Attr>]);
            primitive_builtin_attr_getter_impl!($name, $mnemonic, [<$name Attr>]);
        }
    };

    ($name:ident, $mnemonic:ident, $return_ty:ident) => {
        paste! {
            primitive_builtin_attr_impl!($name, $mnemonic, [<$name Attr>]);
            primitive_builtin_attr_value_impl!($name, $mnemonic, [<$name Attr>], $return_ty);
        }
    };

    ($name:ident, $mnemonic:ident, $return_ty:ident, $get_value_name:ident) => {
        paste! {
            primitive_builtin_attr_impl!($name, $mnemonic, [<$name Attr>]);
            primitive_builtin_attr_value_impl!($name, $mnemonic, [<$name Attr>], $return_ty, $get_value_name);
        }
    };
}

macro_rules! primitive_builtin_attr_impl {
    ($name:ident, $mnemonic:ident, $ty:ident) => {
        #[repr(transparent)]
        #[derive(Copy, Clone)]
        pub struct $ty(AttributeBase);
        impl Attribute for $ty {
            #[inline]
            fn base(&self) -> AttributeBase {
                self.0
            }
        }
        impl TryFrom<AttributeBase> for $ty {
            type Error = InvalidTypeCastError;

            fn try_from(attr: AttributeBase) -> Result<Self, Self::Error> {
                let truth = paste! {
                    unsafe { [<mlir_attr_isa_ $mnemonic>](attr) }
                };
                if truth {
                    Ok(Self(attr))
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
        impl PartialEq<AttributeBase> for $ty {
            fn eq(&self, other: &AttributeBase) -> bool {
                self.0.eq(other)
            }
        }

        paste! {
            primitive_builtin_attr_impl!($ty, $name, $mnemonic, [<mlirAttrIsA $name>]);
        }
    };

    ($ty:ident, $name:ident, $mnemonic:ident, $isa_name:ident) => {
        extern "C" {
            paste! {
                #[link_name = stringify!($isa_name)]
                fn [<mlir_attr_isa_ $mnemonic>](attr: AttributeBase) -> bool;
            }
        }
    };
}

macro_rules! primitive_builtin_attr_getter_impl {
    ($name:ident, $mnemonic:ident, $ty:ident) => {
        paste! {
            primitive_builtin_attr_getter_impl!($name, $mnemonic, $ty, [<mlir $ty Get>]);
        }
    };

    ($name:ident, $mnemonic:ident, $ty:ident, $get_name:ident) => {
        impl $ty {
            pub fn get(context: Context) -> Self {
                paste! {
                    unsafe { [<mlir_ $mnemonic _attr_get>](context) }
                }
            }
        }
        extern "C" {
            paste! {
                #[link_name = stringify!($get_name)]
                fn [<mlir_ $mnemonic _attr_get>](context: Context) -> $ty;
            }
        }
    };
}

macro_rules! primitive_builtin_attr_value_impl {
    ($name:ident, $mnemonic:ident, $ty:ident, $return_ty:ident) => {
        paste! {
            primitive_builtin_attr_value_impl!($name, $mnemonic, $ty, $return_ty, [<mlir $ty GetValue>], [<mlir $ty Get>]);
        }
    };

    ($name:ident, $mnemonic:ident, $ty:ident, $return_ty:ident, $get_value_name:ident) => {
        paste! {
            primitive_builtin_attr_value_impl!($name, $mnemonic, $ty, $return_ty, $get_value_name, [<mlir $ty Get>]);
        }
    };

    ($name:ident, $mnemonic:ident, $ty:ident, $return_ty:ident, $get_value_name:ident, $get_name:ident) => {
        impl $ty {
            pub fn get(context: Context, value: $return_ty) -> Self {
                paste! {
                    unsafe { [<mlir_ $mnemonic _attr_get>](context, value) }
                }
            }

            pub fn value(self) -> $return_ty {
                paste! {
                    unsafe { [<mlir_ $mnemonic _attr_get_value>](self) }
                }
            }
        }
        extern "C" {
            paste! {
                #[link_name = stringify!($get_name)]
                fn [<mlir_ $mnemonic _attr_get>](context: Context, value: $return_ty) -> $ty;
                #[link_name = stringify!($get_value_name)]
                fn [<mlir_ $mnemonic _attr_get_value>](attr: $ty) -> $return_ty;
            }
        }
    }
}

primitive_builtin_attr!(Unit, unit);
primitive_builtin_attr!(Bool, bool, bool);

/// Represents the built-in MLIR attribute which holds a type value
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct TypeAttr(AttributeBase);
impl TypeAttr {
    pub fn get<T: Type>(ty: T) -> Self {
        extern "C" {
            #[link_name = "mlirTypeAttrGet"]
            fn mlir_type_attr_get(ty: TypeBase) -> TypeAttr;
        }
        unsafe { mlir_type_attr_get(ty.base()) }
    }

    pub fn value(&self) -> TypeBase {
        extern "C" {
            #[link_name = "mlirTypeAttrGetValue"]
            fn mlir_type_attr_get_value(attr: AttributeBase) -> TypeBase;
        }
        unsafe { mlir_type_attr_get_value(self.0) }
    }
}
impl Attribute for TypeAttr {
    #[inline]
    fn base(&self) -> AttributeBase {
        self.0
    }
}
impl TryFrom<AttributeBase> for TypeAttr {
    type Error = InvalidTypeCastError;

    fn try_from(attr: AttributeBase) -> Result<Self, Self::Error> {
        extern "C" {
            #[link_name = "mlirAttributeIsAType"]
            fn mlir_attr_isa_type(attr: AttributeBase) -> bool;
        }

        if unsafe { mlir_attr_isa_type(attr) } {
            Ok(Self(attr))
        } else {
            Err(InvalidTypeCastError)
        }
    }
}
impl Display for TypeAttr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.0)
    }
}
impl Eq for TypeAttr {}
impl PartialEq for TypeAttr {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl PartialEq<AttributeBase> for TypeAttr {
    fn eq(&self, other: &AttributeBase) -> bool {
        self.0.eq(other)
    }
}

/// Represents the built-in MLIR integer attribute which holds a variety of different integer types.
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct IntegerAttr(AttributeBase);
impl IntegerAttr {
    pub fn get<T: IntegerLike>(ty: T, value: i64) -> Self {
        unsafe { mlir_integer_attr_get(ty.base(), value) }
    }

    /// Returns the value stored in the attribute as a signless integer that fits in an i64
    pub fn value(self) -> i64 {
        unsafe { mlir_integer_attr_get_value_int(self) }
    }

    /// Returns the value stored in the attribute as an unsigned integer that fits in an u64
    pub fn value_unsigned(self) -> u64 {
        unsafe { mlir_integer_attr_get_value_uint(self) }
    }
}
impl Attribute for IntegerAttr {
    #[inline]
    fn base(&self) -> AttributeBase {
        self.0
    }
}
impl TryFrom<AttributeBase> for IntegerAttr {
    type Error = InvalidTypeCastError;

    fn try_from(attr: AttributeBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_attr_isa_integer(attr) } {
            Ok(Self(attr))
        } else {
            Err(InvalidTypeCastError)
        }
    }
}
impl Display for IntegerAttr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.0)
    }
}
impl Eq for IntegerAttr {}
impl PartialEq for IntegerAttr {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl PartialEq<AttributeBase> for IntegerAttr {
    fn eq(&self, other: &AttributeBase) -> bool {
        self.0.eq(other)
    }
}

extern "C" {
    #[link_name = "mlirIntegerAttrGet"]
    fn mlir_integer_attr_get(ty: TypeBase, value: i64) -> IntegerAttr;
    #[link_name = "mlirIntegerAttrGetValueInt"]
    fn mlir_integer_attr_get_value_int(attr: IntegerAttr) -> i64;
    #[link_name = "mlirIntegerAttrGetValueUInt"]
    fn mlir_integer_attr_get_value_uint(attr: IntegerAttr) -> u64;
    #[link_name = "mlirAttrIsAInteger"]
    fn mlir_attr_isa_integer(attr: AttributeBase) -> bool;
}

/// Represents the built-in MLIR float attribute which holds a variety of floating-point types.
/// In our case, we only ever use it with 64-bit floats.
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct FloatAttr(AttributeBase);
impl FloatAttr {
    pub fn get(ty: TypeBase, value: f64) -> Self {
        unsafe { mlir_float_attr_double_get(ty.context(), ty, value) }
    }

    /// Returns the value stored in the attribute as an f64
    pub fn value(self) -> f64 {
        unsafe { mlir_float_attr_get_value_double(self) }
    }
}
impl Attribute for FloatAttr {
    #[inline]
    fn base(&self) -> AttributeBase {
        self.0
    }
}
impl TryFrom<AttributeBase> for FloatAttr {
    type Error = InvalidTypeCastError;

    fn try_from(attr: AttributeBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_attr_isa_float(attr) } {
            Ok(Self(attr))
        } else {
            Err(InvalidTypeCastError)
        }
    }
}
impl Display for FloatAttr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.0)
    }
}
impl Eq for FloatAttr {}
impl PartialEq for FloatAttr {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl PartialEq<AttributeBase> for FloatAttr {
    fn eq(&self, other: &AttributeBase) -> bool {
        self.0.eq(other)
    }
}

extern "C" {
    #[link_name = "mlirFloatAttrDoubleGet"]
    fn mlir_float_attr_double_get(context: Context, ty: TypeBase, value: f64) -> FloatAttr;
    #[link_name = "mlirFloatAttrGetValueDouble"]
    fn mlir_float_attr_get_value_double(attr: FloatAttr) -> f64;
    #[link_name = "mlirAttrIsAFloat"]
    fn mlir_attr_isa_float(attr: AttributeBase) -> bool;
}

/// Represents the built-in MLIR string attribute, which is used throughout MLIR for identifiers.
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct StringAttr(AttributeBase);
impl StringAttr {
    pub fn get<S: Into<StringRef>>(context: Context, value: S) -> Self {
        unsafe { mlir_string_attr_get(context, value.into()) }
    }

    pub fn get_with_type<S: Into<StringRef>>(value: S, ty: TypeBase) -> Self {
        unsafe { mlir_string_attr_typed_get(ty, value.into()) }
    }

    #[inline]
    pub fn value(self) -> StringRef {
        unsafe { mlir_string_attr_get_value(self) }
    }
}
impl Attribute for StringAttr {
    #[inline]
    fn base(&self) -> AttributeBase {
        self.0
    }
}
impl TryFrom<AttributeBase> for StringAttr {
    type Error = InvalidTypeCastError;

    fn try_from(attr: AttributeBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_attr_isa_string(attr) } {
            Ok(Self(attr))
        } else {
            Err(InvalidTypeCastError)
        }
    }
}
impl Display for StringAttr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.0)
    }
}
impl fmt::Pointer for StringAttr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", &self.0)
    }
}
impl Eq for StringAttr {}
impl PartialEq for StringAttr {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl PartialEq<AttributeBase> for StringAttr {
    fn eq(&self, other: &AttributeBase) -> bool {
        self.0.eq(other)
    }
}

extern "C" {
    #[link_name = "mlirStringAttrGet"]
    fn mlir_string_attr_get(context: Context, name: StringRef) -> StringAttr;
    #[link_name = "mlirStringAttrTypedGet"]
    fn mlir_string_attr_typed_get(ty: TypeBase, value: StringRef) -> StringAttr;
    #[link_name = "mlirAttributeIsAString"]
    fn mlir_attr_isa_string(attr: AttributeBase) -> bool;
    #[link_name = "mlirStringAttrGetValue"]
    fn mlir_string_attr_get_value(attr: StringAttr) -> StringRef;
}

/// Represents the built-in MLIR `SymbolRefAttr`, a potentially nested reference to a symbol
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct SymbolRefAttr(AttributeBase);
impl SymbolRefAttr {
    pub fn get<S: Into<StringRef>>(
        context: Context,
        symbol: S,
        references: &[SymbolRefAttr],
    ) -> Self {
        unsafe {
            mlir_symbol_ref_attr_get(
                context,
                symbol.into(),
                references.len(),
                references.as_ptr(),
            )
        }
    }

    /// Creates a symbol reference attribute in the given context referencing a
    /// symbol identified by the given string inside a list of nested references.
    /// Each of the references in the list must not be nested. The string need not be
    /// null-terminated and its length must be specified.
    #[inline]
    pub fn root_reference(self) -> StringRef {
        unsafe { mlir_symbol_ref_attr_get_root_reference(self) }
    }

    /// Returns the string reference to the leaf referenced symbol. The data remains
    /// live as long as the context in which the attribute lives.
    #[inline]
    pub fn leaf_reference(self) -> StringRef {
        unsafe { mlir_symbol_ref_attr_get_leaf_reference(self) }
    }

    /// Returns the number of references nested in the given symbol reference
    /// attribute.
    #[inline]
    pub fn num_references(self) -> usize {
        unsafe { mlir_symbol_ref_attr_get_num_nested_references(self) }
    }

    /// Returns pos-th reference nested in the given symbol reference attribute.
    pub fn get_nested_reference(self, index: usize) -> SymbolRefAttr {
        let attr = unsafe { mlir_symbol_ref_attr_get_nested_references(self, index) };
        assert!(
            !attr.is_null(),
            "invalid reference index {}, index out of bounds",
            index
        );
        Self(attr)
    }
}
impl Attribute for SymbolRefAttr {
    #[inline]
    fn base(&self) -> AttributeBase {
        self.0
    }
}
impl TryFrom<AttributeBase> for SymbolRefAttr {
    type Error = InvalidTypeCastError;

    fn try_from(attr: AttributeBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_attr_isa_symbol_ref(attr) } {
            Ok(Self(attr))
        } else {
            Err(InvalidTypeCastError)
        }
    }
}
impl Display for SymbolRefAttr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.0)
    }
}
impl fmt::Pointer for SymbolRefAttr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", &self.0)
    }
}
impl Eq for SymbolRefAttr {}
impl PartialEq for SymbolRefAttr {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl PartialEq<AttributeBase> for SymbolRefAttr {
    fn eq(&self, other: &AttributeBase) -> bool {
        self.0.eq(other)
    }
}

extern "C" {
    #[link_name = "mlirAttributeIsASymbolRef"]
    fn mlir_attr_isa_symbol_ref(attr: AttributeBase) -> bool;
    #[link_name = "mlirSymbolRefAttrGet"]
    fn mlir_symbol_ref_attr_get(
        context: Context,
        symbol: StringRef,
        num_references: usize,
        references: *const SymbolRefAttr,
    ) -> SymbolRefAttr;
    #[link_name = "mlirSymbolRefAttrGetRootReference"]
    fn mlir_symbol_ref_attr_get_root_reference(attr: SymbolRefAttr) -> StringRef;
    #[link_name = "mlirSymbolRefAttrGetLeafReference"]
    fn mlir_symbol_ref_attr_get_leaf_reference(attr: SymbolRefAttr) -> StringRef;
    #[link_name = "mlirSymbolRefAttrGetNumNestedReferences"]
    fn mlir_symbol_ref_attr_get_num_nested_references(attr: SymbolRefAttr) -> usize;
    #[link_name = "mlirSymbolRefAttrGetNestedReference"]
    fn mlir_symbol_ref_attr_get_nested_references(
        attr: SymbolRefAttr,
        index: usize,
    ) -> AttributeBase;
}

/// Represents the built-in MLIR `FlatSymbolRefAttr`
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct FlatSymbolRefAttr(AttributeBase);
impl FlatSymbolRefAttr {
    pub fn get<S: Into<StringRef>>(context: Context, symbol: S) -> Self {
        unsafe { mlir_flat_symbol_ref_attr_get(context, symbol.into()) }
    }

    #[inline]
    pub fn value(self) -> StringRef {
        unsafe { mlir_flat_symbol_ref_attr_get_value(self) }
    }
}
impl Attribute for FlatSymbolRefAttr {
    #[inline]
    fn base(&self) -> AttributeBase {
        self.0
    }
}
impl TryFrom<AttributeBase> for FlatSymbolRefAttr {
    type Error = InvalidTypeCastError;

    fn try_from(attr: AttributeBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_attr_isa_flat_symbol_ref(attr) } {
            Ok(Self(attr))
        } else {
            Err(InvalidTypeCastError)
        }
    }
}
impl Display for FlatSymbolRefAttr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.0)
    }
}
impl fmt::Pointer for FlatSymbolRefAttr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", &self.0)
    }
}
impl Eq for FlatSymbolRefAttr {}
impl PartialEq for FlatSymbolRefAttr {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl PartialEq<AttributeBase> for FlatSymbolRefAttr {
    fn eq(&self, other: &AttributeBase) -> bool {
        self.0.eq(other)
    }
}

extern "C" {
    #[link_name = "mlirAttributeIsAFlatSymbolRef"]
    fn mlir_attr_isa_flat_symbol_ref(attr: AttributeBase) -> bool;
    #[link_name = "mlirFlatSymbolRefAttrGet"]
    fn mlir_flat_symbol_ref_attr_get(context: Context, symbol: StringRef) -> FlatSymbolRefAttr;
    #[link_name = "mlirFlatSymbolRefAttrGetValue"]
    fn mlir_flat_symbol_ref_attr_get_value(attr: FlatSymbolRefAttr) -> StringRef;
}

/// Represents the built-in MLIR `DictionaryAttr`
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct DictionaryAttr(AttributeBase);
impl DictionaryAttr {
    /// Gets a new DictionaryAttr with the given elements
    pub fn get(context: Context, elements: &[NamedAttribute]) -> Self {
        unsafe { mlir_dictionary_attr_get(context, elements.len(), elements.as_ptr()) }
    }

    /// Returns the number of attributes contained in a dictionary attribute.
    pub fn len(self) -> usize {
        unsafe { mlir_dictionary_attr_get_num_elements(self) }
    }

    /// Returns pos-th element of the given dictionary attribute.
    pub fn get_element(self, index: usize) -> NamedAttribute {
        let attr = unsafe { mlir_dictionary_attr_get_element(self, index) };
        assert!(
            !attr.is_null(),
            "invalid dictionary index {}, out of bounds",
            index
        );
        attr
    }

    /// Returns the dictionary attribute element with the given name or NULL if the
    /// given name does not exist in the dictionary.
    pub fn get_element_by_name<S: Into<StringRef>>(self, name: S) -> Option<AttributeBase> {
        let attr = unsafe { mlir_dictionary_attr_get_element_by_name(self, name.into()) };
        if attr.is_null() {
            None
        } else {
            Some(attr)
        }
    }
}
impl Attribute for DictionaryAttr {
    #[inline]
    fn base(&self) -> AttributeBase {
        self.0
    }
}
impl TryFrom<AttributeBase> for DictionaryAttr {
    type Error = InvalidTypeCastError;

    fn try_from(attr: AttributeBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_attr_isa_dictionary(attr) } {
            Ok(Self(attr))
        } else {
            Err(InvalidTypeCastError)
        }
    }
}
impl Display for DictionaryAttr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.0)
    }
}
impl fmt::Pointer for DictionaryAttr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", &self.0)
    }
}
impl Eq for DictionaryAttr {}
impl PartialEq for DictionaryAttr {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl PartialEq<AttributeBase> for DictionaryAttr {
    fn eq(&self, other: &AttributeBase) -> bool {
        self.0.eq(other)
    }
}

extern "C" {
    #[link_name = "mlirAttributeIsADictionary"]
    fn mlir_attr_isa_dictionary(attr: AttributeBase) -> bool;
    #[link_name = "mlirDictionaryAttrGet"]
    fn mlir_dictionary_attr_get(
        context: Context,
        num_elements: usize,
        elements: *const NamedAttribute,
    ) -> DictionaryAttr;
    #[link_name = "mlirDictionaryAttrGetNumElements"]
    fn mlir_dictionary_attr_get_num_elements(attr: DictionaryAttr) -> usize;
    #[link_name = "mlirDictionaryAttrGetElement"]
    fn mlir_dictionary_attr_get_element(attr: DictionaryAttr, index: usize) -> NamedAttribute;
    #[link_name = "mlirDictionaryAttrGetElementByName"]
    fn mlir_dictionary_attr_get_element_by_name(
        attr: DictionaryAttr,
        name: StringRef,
    ) -> AttributeBase;
}

/// Represents the built-in MLIR `AffineMapAttr`
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct AffineMapAttr(AttributeBase);
impl AffineMapAttr {
    /// Get a new AffineMapAttr containing the given AffineMap
    pub fn get(map: AffineMap) -> Self {
        unsafe { mlir_affine_map_attr_get(map) }
    }

    /// Get the AffineMap value
    pub fn value(&self) -> AffineMap {
        unsafe { mlir_affine_map_attr_get_value(*self) }
    }
}
impl Attribute for AffineMapAttr {
    #[inline]
    fn base(&self) -> AttributeBase {
        self.0
    }
}
impl TryFrom<AttributeBase> for AffineMapAttr {
    type Error = InvalidTypeCastError;

    fn try_from(attr: AttributeBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_attr_isa_affine_map(attr) } {
            Ok(Self(attr))
        } else {
            Err(InvalidTypeCastError)
        }
    }
}
impl Display for AffineMapAttr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.0)
    }
}
impl fmt::Pointer for AffineMapAttr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", &self.0)
    }
}
impl Eq for AffineMapAttr {}
impl PartialEq for AffineMapAttr {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl PartialEq<AttributeBase> for AffineMapAttr {
    fn eq(&self, other: &AttributeBase) -> bool {
        self.0.eq(other)
    }
}

extern "C" {
    #[link_name = "mlirAttributeIsAAffineMap"]
    fn mlir_attr_isa_affine_map(attr: AttributeBase) -> bool;
    #[link_name = "mlirAffineMapAttrGet"]
    fn mlir_affine_map_attr_get(map: AffineMap) -> AffineMapAttr;
    #[link_name = "mlirAffineMapAttrGetValue"]
    fn mlir_affine_map_attr_get_value(attr: AffineMapAttr) -> AffineMap;
}

/// Represents the built-in MLIR `ArrayAttr`
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ArrayAttr(AttributeBase);
impl ArrayAttr {
    /// Gets a new DictionaryAttr with the given elements
    pub fn get(context: Context, elements: &[AttributeBase]) -> Self {
        unsafe { mlir_array_attr_get(context, elements.len(), elements.as_ptr()) }
    }

    /// Returns the number of attributes contained in an array attribute.
    pub fn len(self) -> usize {
        unsafe { mlir_array_attr_get_num_elements(self) }
    }

    /// Returns pos-th element of the given array attribute.
    pub fn get_element(self, index: usize) -> AttributeBase {
        let attr = unsafe { mlir_array_attr_get_element(self, index) };
        assert!(
            !attr.is_null(),
            "invalid array index {}, out of bounds",
            index
        );
        attr
    }
}
impl Attribute for ArrayAttr {
    #[inline]
    fn base(&self) -> AttributeBase {
        self.0
    }
}
impl TryFrom<AttributeBase> for ArrayAttr {
    type Error = InvalidTypeCastError;

    fn try_from(attr: AttributeBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_attr_isa_array(attr) } {
            Ok(Self(attr))
        } else {
            Err(InvalidTypeCastError)
        }
    }
}
impl Display for ArrayAttr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.0)
    }
}
impl fmt::Pointer for ArrayAttr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", &self.0)
    }
}
impl Eq for ArrayAttr {}
impl PartialEq for ArrayAttr {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl PartialEq<AttributeBase> for ArrayAttr {
    fn eq(&self, other: &AttributeBase) -> bool {
        self.0.eq(other)
    }
}

extern "C" {
    #[link_name = "mlirAttributeIsAArray"]
    fn mlir_attr_isa_array(attr: AttributeBase) -> bool;
    #[link_name = "mlirArrayAttrGet"]
    fn mlir_array_attr_get(
        context: Context,
        num_elements: usize,
        elements: *const AttributeBase,
    ) -> ArrayAttr;
    #[link_name = "mlirArrayAttrGetNumElements"]
    fn mlir_array_attr_get_num_elements(attr: ArrayAttr) -> usize;
    #[link_name = "mlirArrayAttrGetElement"]
    fn mlir_array_attr_get_element(attr: ArrayAttr, index: usize) -> AttributeBase;
}
