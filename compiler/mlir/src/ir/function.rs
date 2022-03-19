use super::*;

/// Represents the built-in FuncOp from the `func` dialect
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct FuncOp(OperationBase);
impl FuncOp {
    /// Returns the function symbol name
    pub fn name(self) -> StringRef {
        SymbolTable::get_symbol_name(self)
    }

    /// Returns the visibility of this function
    pub fn visibility(self) -> Visibility {
        SymbolTable::get_symbol_visibility(self)
    }

    /// Sets the visibility of this function
    pub fn set_visibility(self, visibility: Visibility) {
        SymbolTable::set_symbol_visibility(self, visibility)
    }

    /// Returns the type signature of this function
    pub fn get_type(&self) -> FunctionType {
        unsafe { mlir_func_op_get_type(*self) }
    }
}
impl TryFrom<OperationBase> for FuncOp {
    type Error = InvalidTypeCastError;

    fn try_from(op: OperationBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_operation_isa_func_op(op) } {
            Ok(Self(op))
        } else {
            Err(InvalidTypeCastError)
        }
    }
}
impl Operation for FuncOp {
    fn base(&self) -> OperationBase {
        self.0
    }
}

extern "C" {
    #[link_name = "mlirOperationIsAFuncOp"]
    fn mlir_operation_isa_func_op(op: OperationBase) -> bool;
    #[link_name = "mlirFuncOpGetType"]
    fn mlir_func_op_get_type(op: FuncOp) -> FunctionType;
    #[link_name = "mlirFuncBuildFuncOp"]
    pub(super) fn mlir_func_build_func_op(
        builder: OpBuilderBase,
        loc: Location,
        name: StringRef,
        ty: types::FunctionType,
        num_attrs: usize,
        attrs: *const NamedAttribute,
        num_arg_attrs: usize,
        arg_attrs: *const DictionaryAttr,
    ) -> FuncOp;
}
