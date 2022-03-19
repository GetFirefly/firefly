use crate::{AddressSpace, Context, Type, TypeBase, Variadic};

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
        unsafe { mlir_llvm_void_type_get(context) }
    }
}

#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct PointerType<T: LlvmType>(TypeBase);
impl<T: LlvmType> Type for PointerType<T> {
    #[inline]
    fn base(&self) -> TypeBase {
        self.0
    }
}
impl<T: LlvmType> LlvmType for PointerType<T> {}
impl<T: LlvmType> PointerType<T> {
    /// Returns a pointer to a value of type `pointee`, in the given address space
    pub fn get(pointee: T, address_space: AddressSpace) -> Self {
        Self(unsafe { mlir_llvm_pointer_type_get(pointee.base(), address_space) })
    }
}

/// An array type of a specific length and element type
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ArrayType<T: LlvmType>(TypeBase);
impl<T: LlvmType> LlvmType for ArrayType<T> {}
impl<T: LlvmType> Type for ArrayType<T> {
    #[inline]
    fn base(&self) -> TypeBase {
        self.0
    }
}
impl<T: LlvmType> ArrayType<T> {
    pub fn get(element_type: T) -> Self {
        Self(unsafe { mlir_llvm_array_type_get(element_type.base()) })
    }
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
                self.context(),
                fields.len(),
                fields.as_ptr(),
                /*packed=*/ false,
            )
        }
    }

    pub fn get_packed(context: Context, fields: &[TypeBase]) -> Self {
        unsafe {
            mlir_llvm_struct_type_get(
                self.context(),
                fields.len(),
                fields.as_ptr(),
                /*packed=*/ true,
            )
        }
    }
}

extern "C" {
    #[link_name = "mlirLLVMPointerTypeGet"]
    fn mlir_llvm_pointer_type_get(pointee: TypeBase, address_space: AddressSpace) -> TypeBase;
    #[link_name = "mlirLLVMVoidTypeGet"]
    fn mlir_llvm_void_type_get(context: Context) -> VoidType;
    #[link_name = "mlirLLVMArrayTypeGet"]
    fn mlir_llvm_array_type_get(elementType: TypeBase, arity: u32) -> TypeBase;
    #[link_name = "mlirLLVMFunctionTypeGet"]
    fn mlir_llvm_function_type_get(
        resultType: TypeBase,
        argc: usize,
        argv: *const TypeBase,
        is_vararg: Variadic,
    ) -> FunctionType;
    #[link_name = "mlirLLVMStructTypeGet"]
    fn mlir_llvm_struct_type_get(
        context: Context,
        num_fields: usize,
        fields: *const TypeBase,
        packed: bool,
    ) -> StructType;
}
