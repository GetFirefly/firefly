use super::*;

extern "C" {
    type LlvmValueUse;
}

/// Represents the use of a value
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ValueUse(*const LlvmValueUse);
impl ValueUse {
    #[inline(always)]
    fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Gets the user this use corresponds to
    pub fn user(self) -> User {
        extern "C" {
            fn LLVMGetUser(vu: ValueUse) -> User;
        }
        unsafe { LLVMGetUser(self) }
    }

    /// Gets the value this use corresponds to
    pub fn value(self) -> ValueBase {
        extern "C" {
            fn LLVMGetUsedValue(vu: ValueUse) -> ValueBase;
        }
        unsafe { LLVMGetUsedValue(self) }
    }
}

/// Represents a user of a value
///
/// Subtypes of this include:
///
/// * constants
/// * instructions
/// * operators
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct User(ValueBase);
impl User {
    /// Returns the number of operands in this value
    pub fn num_operands(self) -> usize {
        extern "C" {
            fn LLVMGetNumOperands(user: User) -> i32;
        }
        unsafe { LLVMGetNumOperands(self) }.try_into().unwrap()
    }

    /// Gets the operand at the given index in this user
    pub fn operand(self, index: usize) -> ValueBase {
        extern "C" {
            fn LLVMGetOperand(user: User, index: u32) -> ValueBase;
        }
        unsafe { LLVMGetOperand(self, index.try_into().unwrap()) }
    }

    /// Get the use of an operand at the given index in this user
    pub fn operand_use(self, index: usize) -> ValueUse {
        extern "C" {
            fn LLVMGetOperandUse(user: User, index: u32) -> ValueUse;
        }
        unsafe { LLVMGetOperandUse(self, index.try_into().unwrap()) }
    }

    /// Sets the operand at the given index to `value`
    pub fn set_operand(self, index: usize, value: ValueBase) {
        extern "C" {
            fn LLVMSetOperand(user: User, index: u32, value: ValueBase);
        }
        unsafe { LLVMSetOperand(self, index.try_into().unwrap(), value) }
    }
}
impl TryFrom<ValueBase> for User {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::ConstantExpr
            | ValueKind::ConstantArray
            | ValueKind::ConstantStruct
            | ValueKind::ConstantVector
            | ValueKind::ConstantAggregateZero
            | ValueKind::ConstantDataArray
            | ValueKind::ConstantDataVector
            | ValueKind::ConstantInt
            | ValueKind::ConstantFP
            | ValueKind::ConstantPointerNull
            | ValueKind::ConstantTokenNone
            | ValueKind::Instruction => Ok(Self(value)),
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// An iterator over uses of a value
pub struct ValueUseIter(ValueUse);
impl ValueUseIter {
    pub(super) fn new(value: ValueBase) -> Self {
        extern "C" {
            fn LLVMGetFirstUse(value: ValueBase) -> ValueUse;
        }
        Self(unsafe { LLVMGetFirstUse(value) })
    }
}
impl Iterator for ValueUseIter {
    type Item = ValueUse;

    fn next(&mut self) -> Option<Self::Item> {
        extern "C" {
            fn LLVMGetNextUse(vu: ValueUse) -> ValueUse;
        }
        if self.0.is_null() {
            return None;
        }
        let next = self.0;
        self.0 = unsafe { LLVMGetNextUse(next) };
        Some(next)
    }
}
impl std::iter::FusedIterator for ValueUseIter {}
