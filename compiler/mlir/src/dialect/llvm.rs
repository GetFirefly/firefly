use std::ops::Deref;

use liblumen_llvm::{Linkage, StringRef};

use crate::{AddressSpace, Context, Location, SymbolTable, Variadic};
use crate::{Attribute, AttributeBase, NamedAttribute};
use crate::{Builder, BuilderBase, OpBuilder};
use crate::{InvalidTypeCastError, OwnedRegion};
use crate::{Operation, OperationBase, OperationState};
use crate::{Type, TypeBase};

/// Primary builder for the LLVM dialect
///
/// Wraps mlir::OpBuilder and provides functionality for constructing dialect operations, types, and attributes
#[derive(Copy, Clone)]
pub struct LlvmBuilder<'a, B: OpBuilder> {
    builder: &'a B,
}
impl<'a, B: OpBuilder> LlvmBuilder<'a, B> {
    pub fn new(builder: &'a B) -> Self {
        Self { builder }
    }
}
impl<'a, B: OpBuilder> Builder for LlvmBuilder<'a, B> {
    #[inline]
    fn base(&self) -> BuilderBase {
        self.builder.base()
    }
}
impl<'a, B: OpBuilder> OpBuilder for LlvmBuilder<'a, B> {}
impl<'a, B: OpBuilder> Deref for LlvmBuilder<'a, B> {
    type Target = B;

    fn deref(&self) -> &Self::Target {
        self.builder
    }
}

//----------------------------
// Types
//----------------------------

/// Marker type for LLVM types
pub trait LlvmType: Type {}

/// A type which represents nothing, e.g. Rust's `()` type, or C's `void` type
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct VoidType(TypeBase);
impl LlvmType for VoidType {}
impl Type for VoidType {
    #[inline]
    fn base(&self) -> TypeBase {
        self.0
    }
}
impl VoidType {
    pub fn get(context: Context) -> Self {
        extern "C" {
            #[link_name = "mlirLLVMVoidTypeGet"]
            fn mlir_llvm_void_type_get(context: Context) -> VoidType;
        }
        unsafe { mlir_llvm_void_type_get(context) }
    }
}
impl<'a, B: OpBuilder> LlvmBuilder<'a, B> {
    #[inline]
    pub fn get_void_type(&self) -> VoidType {
        VoidType::get(self.context())
    }
}

#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct PointerType(TypeBase);
impl Type for PointerType {
    #[inline]
    fn base(&self) -> TypeBase {
        self.0
    }
}
impl LlvmType for PointerType {}
impl PointerType {
    /// Returns a pointer to a value of type `pointee`, in the given address space
    pub fn get<T: Type>(pointee: T, address_space: AddressSpace) -> Self {
        Self(unsafe { mlir_llvm_pointer_type_get(pointee.base(), address_space) })
    }

    pub fn get_unchecked(pointee: TypeBase, address_space: AddressSpace) -> Self {
        Self(unsafe { mlir_llvm_pointer_type_get(pointee, address_space) })
    }
}
impl<'a, B: OpBuilder> LlvmBuilder<'a, B> {
    #[inline]
    pub fn get_pointer_type<T: Type>(
        &self,
        pointee: T,
        address_space: AddressSpace,
    ) -> PointerType {
        PointerType::get(pointee, address_space)
    }
}
extern "C" {
    #[link_name = "mlirLLVMPointerTypeGet"]
    fn mlir_llvm_pointer_type_get(pointee: TypeBase, address_space: AddressSpace) -> TypeBase;
}

/// An array type of a specific length and element type
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ArrayType(TypeBase);
impl LlvmType for ArrayType {}
impl Type for ArrayType {
    #[inline]
    fn base(&self) -> TypeBase {
        self.0
    }
}
impl ArrayType {
    pub fn get<T: Type>(element_type: T, arity: usize) -> Self {
        Self(unsafe { mlir_llvm_array_type_get(element_type.base(), arity.try_into().unwrap()) })
    }

    pub fn get_unchecked(element_type: TypeBase, arity: usize) -> Self {
        Self(unsafe { mlir_llvm_array_type_get(element_type, arity.try_into().unwrap()) })
    }
}
impl<'a, B: OpBuilder> LlvmBuilder<'a, B> {
    #[inline]
    pub fn get_array_type<T: Type>(&self, element_type: T, arity: usize) -> ArrayType {
        ArrayType::get(element_type, arity)
    }
}

extern "C" {
    #[link_name = "mlirLLVMArrayTypeGet"]
    fn mlir_llvm_array_type_get(elementType: TypeBase, arity: u32) -> TypeBase;
}

/// A struct type is a set of heterogenous fields laid out in memory in order, potentially packed
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct StructType(TypeBase);
impl LlvmType for StructType {}
impl Type for StructType {
    #[inline]
    fn base(&self) -> TypeBase {
        self.0
    }
}
impl StructType {
    pub fn get(context: Context, fields: &[TypeBase]) -> Self {
        unsafe {
            mlir_llvm_struct_type_get(
                context,
                fields.len(),
                fields.as_ptr(),
                /*packed=*/ false,
            )
        }
    }

    pub fn get_packed(context: Context, fields: &[TypeBase]) -> Self {
        unsafe {
            mlir_llvm_struct_type_get(
                context,
                fields.len(),
                fields.as_ptr(),
                /*packed=*/ true,
            )
        }
    }
}
impl<'a, B: OpBuilder> LlvmBuilder<'a, B> {
    #[inline]
    pub fn get_struct_type(&self, fields: &[TypeBase], packed: bool) -> StructType {
        if packed {
            StructType::get_packed(self.context(), fields)
        } else {
            StructType::get(self.context(), fields)
        }
    }
}

extern "C" {
    #[link_name = "mlirLLVMStructTypeLiteralGet"]
    fn mlir_llvm_struct_type_get(
        context: Context,
        num_fields: usize,
        fields: *const TypeBase,
        packed: bool,
    ) -> StructType;
}

/// A function type composed of the given result and parameter types
///
/// To represent a function that has no result, use `VoidType`
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct FunctionType(TypeBase);
impl LlvmType for FunctionType {}
impl Type for FunctionType {
    #[inline]
    fn base(&self) -> TypeBase {
        self.0
    }
}
impl FunctionType {
    pub fn get(result: TypeBase, params: &[TypeBase], variadic: Variadic) -> Self {
        unsafe { mlir_llvm_function_type_get(result, params.len(), params.as_ptr(), variadic) }
    }
}
impl<'a, B: OpBuilder> LlvmBuilder<'a, B> {
    #[inline]
    pub fn get_function_type<T: Type>(
        &self,
        result: T,
        params: &[TypeBase],
        variadic: Variadic,
    ) -> FunctionType {
        FunctionType::get(result.base(), params, variadic)
    }
}

extern "C" {
    #[link_name = "mlirLLVMFunctionTypeGet"]
    fn mlir_llvm_function_type_get(
        resultType: TypeBase,
        argc: usize,
        argv: *const TypeBase,
        is_vararg: Variadic,
    ) -> FunctionType;
}

//----------------------------
// Attributes
//----------------------------

/// LinkageAttr is used to represent LLVM IR linkage
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct LinkageAttr(AttributeBase);
impl LinkageAttr {
    #[inline]
    pub fn get(context: Context, value: Linkage) -> Self {
        extern "C" {
            fn mlirLLVMLinkageAttrGet(context: Context, name: StringRef) -> LinkageAttr;
        }
        let name = match value {
            Linkage::Private => "private",
            Linkage::Internal => "internal",
            Linkage::AvailableExternally => "available_externally",
            Linkage::LinkOnceAny => "linkonce",
            Linkage::WeakAny => "weak",
            Linkage::Common => "common",
            Linkage::Appending => "appending",
            Linkage::ExternalWeak => "extern_weak",
            Linkage::LinkOnceODR => "linkonce_odr",
            Linkage::WeakODR => "weak_odr",
            Linkage::External => "external",
            other => panic!(
                "unsupported linkage type with mlir LinkageAttr: {:?}",
                other
            ),
        };
        unsafe { mlirLLVMLinkageAttrGet(context, name.into()) }
    }
}
impl Attribute for LinkageAttr {
    #[inline]
    fn base(&self) -> AttributeBase {
        self.0
    }
}
impl<'a, B: OpBuilder> LlvmBuilder<'a, B> {
    #[inline]
    pub fn get_linkage_attr(&self, linkage: Linkage) -> LinkageAttr {
        LinkageAttr::get(self.context(), linkage)
    }
}

//----------------------------
// Operations
//----------------------------

/// Represents a function in LLVM dialect
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct FuncOp(OperationBase);
impl Operation for FuncOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl TryFrom<OperationBase> for FuncOp {
    type Error = InvalidTypeCastError;

    #[inline]
    fn try_from(op: OperationBase) -> Result<Self, Self::Error> {
        extern "C" {
            fn mlirLLVMFuncOpIsA(op: OperationBase) -> bool;
        }
        if unsafe { mlirLLVMFuncOpIsA(op) } {
            Ok(Self(op))
        } else {
            Err(InvalidTypeCastError)
        }
    }
}
impl<'a, B: OpBuilder> LlvmBuilder<'a, B> {
    #[inline]
    pub fn build_func<S: Into<StringRef>>(
        &self,
        loc: Location,
        name: S,
        ty: FunctionType,
        linkage: Linkage,
        attrs: &[NamedAttribute],
    ) -> FuncOp {
        let name = name.into();
        let mut state = OperationState::get("llvm.func", loc);
        let regions = vec![OwnedRegion::new().release()];
        state.add_owned_regions(regions.as_slice());
        let mut base_attrs = Vec::with_capacity(3);
        base_attrs.push(self.get_named_attr(
            SymbolTable::get_symbol_attr_name(),
            self.get_string_attr(name),
        ));
        base_attrs.push(self.get_named_attr("function_type", self.get_type_attr(ty)));
        base_attrs.push(self.get_named_attr("linkage", self.get_linkage_attr(linkage)));
        state.add_attributes(base_attrs.as_slice());
        state.add_attributes(attrs);

        self.create_operation(state).unwrap()
    }
}
