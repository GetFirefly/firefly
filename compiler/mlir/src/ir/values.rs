use std::ffi::c_void;
use std::fmt::{self, Display};

use crate::support::{self, MlirStringCallback};

use super::*;

extern "C" {
    type MlirValue;
}

/// Represents a value of any provenance in Rust
///
/// A value can originate either as a block argument or as a result of an operation.
///
/// In most cases we don't actually care what type of value it is, so APIs should
/// either use this trait, or the `ValueBase` type for generic operations.
pub trait Value {
    /// Returns the type of this value
    fn get_type(&self) -> TypeBase {
        unsafe { mlir_value_get_type(self.base()) }
    }
    /// Dumps the textual representation of this value to stderr
    fn dump(&self) {
        unsafe { mlir_value_dump(self.base()) }
    }
    /// Returns this value as a ValueBase
    fn base(&self) -> ValueBase;
}

/// Represents a value of any provenance in the FFI bridge
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ValueBase(*mut MlirValue);
impl Value for ValueBase {
    #[inline(always)]
    fn base(&self) -> ValueBase {
        *self
    }
}
impl Default for ValueBase {
    fn default() -> ValueBase {
        Self(std::ptr::null_mut::<usize>() as *mut MlirValue)
    }
}
impl ValueBase {
    #[inline(always)]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Returns true if this value is of the given concrete type
    #[inline(always)]
    pub fn isa<T>(self) -> bool
    where
        T: TryFrom<ValueBase>,
    {
        T::try_from(self).is_ok()
    }

    /// Tries to cast this value to the given concrete type
    #[inline(always)]
    pub fn dyn_cast<T>(self) -> Result<T, InvalidTypeCastError>
    where
        T: TryFrom<ValueBase, Error = InvalidTypeCastError>,
    {
        T::try_from(self)
    }
}
impl Eq for ValueBase {}
impl PartialEq for ValueBase {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlir_value_equal(*self, *other) }
    }
}
impl fmt::Pointer for ValueBase {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", self.0)
    }
}
impl Display for ValueBase {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe {
            mlir_value_print(
                *self,
                support::write_to_formatter,
                f as *mut _ as *mut c_void,
            );
        }
        Ok(())
    }
}

/// Represents the value of a block argument
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct BlockArgument(ValueBase);
impl BlockArgument {
    #[inline(always)]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Returns the block in which this value is defined as an argument.
    pub fn owner(self) -> Block {
        unsafe { mlir_block_argument_get_owner(self) }
    }

    /// Returns the position of the value in the argument list of its block.
    pub fn position(self) -> usize {
        unsafe { mlir_block_argument_get_arg_number(self) }
    }

    /// Sets the type of the block argument to the given type.
    pub fn set_type(self, ty: TypeBase) {
        unsafe { mlir_block_argument_set_type(self, ty) }
    }
}
impl Value for BlockArgument {
    #[inline]
    fn base(&self) -> ValueBase {
        self.0
    }
}
impl TryFrom<ValueBase> for BlockArgument {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_value_isa_block_argument(value) } {
            Ok(Self(value))
        } else {
            Err(InvalidTypeCastError)
        }
    }
}
impl Display for BlockArgument {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl fmt::Pointer for BlockArgument {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", self.0)
    }
}
impl Eq for BlockArgument {}
impl PartialEq for BlockArgument {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl PartialEq<ValueBase> for BlockArgument {
    #[inline]
    fn eq(&self, other: &ValueBase) -> bool {
        self.0.eq(other)
    }
}

/// Represents a value produced as the result of an operation
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct OpResult(ValueBase);
impl OpResult {
    #[inline(always)]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Returns the operation that introduced this value.
    pub fn owner(self) -> OperationBase {
        unsafe { mlir_op_result_get_owner(self) }
    }

    /// Returns the position of the value in the result set of its defining operation
    pub fn position(self) -> usize {
        unsafe { mlir_op_result_get_result_number(self) }
    }
}
impl TryFrom<ValueBase> for OpResult {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_value_isa_op_result(value) } {
            Ok(Self(value))
        } else {
            Err(InvalidTypeCastError)
        }
    }
}
impl Value for OpResult {
    #[inline]
    fn base(&self) -> ValueBase {
        self.0
    }
}
impl Display for OpResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl fmt::Pointer for OpResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", self.0)
    }
}
impl Eq for OpResult {}
impl PartialEq for OpResult {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl PartialEq<ValueBase> for OpResult {
    #[inline]
    fn eq(&self, other: &ValueBase) -> bool {
        self.0.eq(other)
    }
}

extern "C" {
    #[link_name = "mlirValueEqual"]
    fn mlir_value_equal(a: ValueBase, b: ValueBase) -> bool;
    #[link_name = "mlirValueIsABlockArgument"]
    fn mlir_value_isa_block_argument(value: ValueBase) -> bool;
    #[link_name = "mlirValueIsAOpResult"]
    fn mlir_value_isa_op_result(value: ValueBase) -> bool;
    #[link_name = "mlirBlockArgumentGetOwner"]
    fn mlir_block_argument_get_owner(value: BlockArgument) -> Block;
    #[link_name = "mlirBlockArgumentGetArgNumber"]
    fn mlir_block_argument_get_arg_number(value: BlockArgument) -> usize;
    #[link_name = "mlirBlockArgumentSetType"]
    fn mlir_block_argument_set_type(value: BlockArgument, ty: TypeBase);
    #[link_name = "mlirOpResultGetOwner"]
    fn mlir_op_result_get_owner(value: OpResult) -> OperationBase;
    #[link_name = "mlirOpResultGetResultNumber"]
    fn mlir_op_result_get_result_number(value: OpResult) -> usize;
    #[link_name = "mlirValueGetType"]
    fn mlir_value_get_type(value: ValueBase) -> TypeBase;
    #[link_name = "mlirValueDump"]
    fn mlir_value_dump(value: ValueBase);
    #[link_name = "mlirValuePrint"]
    fn mlir_value_print(value: ValueBase, callback: MlirStringCallback, userdata: *const c_void);
}
