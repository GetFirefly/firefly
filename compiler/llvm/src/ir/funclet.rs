use super::*;
use crate::support::StringRef;

extern "C" {
    type LlvmOperandBundle;
}

/// A funclet pad is an SEH exception handling landing pad, and has some common functionality
pub trait FuncletPad: Instruction {
    /// Returns the number of argument operands given to this call
    fn num_arguments(&self) -> usize {
        extern "C" {
            fn LLVMGetNumArgOperands(inst: ValueBase) -> u32;
        }
        unsafe { LLVMGetNumArgOperands(self.base()) as usize }
    }

    fn get_argument(&self, index: usize) -> ValueBase {
        extern "C" {
            fn LLVMGetArgOperand(funclet: ValueBase, index: u32) -> ValueBase;
        }
        unsafe { LLVMGetArgOperand(self.base(), index.try_into().unwrap()) }
    }

    fn set_argument(&self, index: usize, value: ValueBase) {
        extern "C" {
            fn LLVMSetArgOperand(funclet: ValueBase, index: u32, value: ValueBase);
        }
        unsafe { LLVMSetArgOperand(self.base(), index.try_into().unwrap(), value) }
    }
}
impl FuncletPad for CatchSwitchInst {}
impl FuncletPad for CatchPadInst {}
impl FuncletPad for CleanupPadInst {}

/// A structure representing an active landing pad for the duration of a basic
/// block.
///
/// Each `Block` may contain an instance of this, indicating whether the block
/// is part of a landing pad or not. This is used to make decision about whether
/// to emit `invoke` instructions (e.g., in a landing pad we don't continue to
/// use `invoke`) and also about various function call metadata.
///
/// For GNU exceptions (`landingpad` + `resume` instructions) this structure is
/// just a bunch of `None` instances (not too interesting), but for MSVC
/// exceptions (`cleanuppad` + `cleanupret` instructions) this contains data.
/// When inside of a landing pad, each function call in LLVM IR needs to be
/// annotated with which landing pad it's a part of. This is accomplished via
/// the `OperandBundle` value created for MSVC landing pads.
pub struct Funclet {
    pad: ValueBase,
    operand: OwnedOperandBundle,
}
/// A funclet when used as a value is referring to the pad itself, not the operand bundle metadata
impl Value for Funclet {
    #[inline]
    fn base(&self) -> ValueBase {
        self.pad
    }
}
impl Instruction for Funclet {}
impl FuncletPad for Funclet {}
impl Funclet {
    pub fn new(pad: ValueBase) -> Self {
        Self {
            pad,
            operand: OperandBundle::new("funclet", &[pad]),
        }
    }

    pub fn pad(&self) -> ValueBase {
        self.pad
    }

    /// Returns a borrowed reference to the operand bundle for this funclet
    pub fn bundle(&self) -> OperandBundle {
        self.operand.0
    }
}

/// Represents a borrowed reference to an operand bundle
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct OperandBundle(*const LlvmOperandBundle);
impl OperandBundle {
    /// Returns an empty operand bundle reference
    ///
    /// This is used in the FFI bridge for optional parameters
    pub const fn null() -> Self {
        let ptr =
            unsafe { std::mem::transmute::<*const (), *const LlvmOperandBundle>(std::ptr::null()) };
        Self(ptr)
    }

    pub fn new<S: Into<StringRef>>(name: S, operands: &[ValueBase]) -> OwnedOperandBundle {
        extern "C" {
            fn LLVMFireflyBuildOperandBundle(
                name: *const u8,
                len: usize,
                operands: *const ValueBase,
                num_operands: u32,
            ) -> OwnedOperandBundle;
        }

        let name = name.into();
        unsafe {
            LLVMFireflyBuildOperandBundle(
                name.data,
                name.len,
                operands.as_ptr(),
                operands.len().try_into().unwrap(),
            )
        }
    }
}

/// Represents an owned reference to an operand bundle
#[repr(transparent)]
pub struct OwnedOperandBundle(OperandBundle);
impl Drop for OwnedOperandBundle {
    fn drop(&mut self) {
        extern "C" {
            fn LLVMFireflyFreeOperandBundle(bundle: OperandBundle);
        }

        unsafe {
            LLVMFireflyFreeOperandBundle(self.0);
        }
    }
}
