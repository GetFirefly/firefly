use super::*;
use crate::support::*;

extern "C" {
    type LlvmBlock;
}

/// Represents a block in a function
///
/// A block is a sequence of one or more instructions, with the invariant
/// that there is only one branching/terminator instruction in the block,
/// and that instruction must the the last one in the block.
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Block(*const LlvmBlock);
impl Block {
    /// Returns a null Block instance, used for optional values in the FFI bridge
    pub const fn null() -> Block {
        let ptr = unsafe { std::mem::transmute::<*const (), *const LlvmBlock>(std::ptr::null()) };
        Self(ptr)
    }

    #[inline(always)]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Return the name of this block
    pub fn name(self) -> StringRef {
        extern "C" {
            fn LLVMBasicBlockName(bb: Block) -> *const std::os::raw::c_char;
        }
        unsafe { StringRef::from_ptr(LLVMBasicBlockName(self)) }
    }

    /// Return the containing function
    pub fn function(self) -> Function {
        extern "C" {
            fn LLVMGetBasicBlockParent(bb: Block) -> Function;
        }
        unsafe { LLVMGetBasicBlockParent(self) }
    }

    /// Get this block as an llvm::Value
    pub fn as_value(self) -> BlockValue {
        extern "C" {
            fn LLVMBasicBlockAsValue(bb: Block) -> BlockValue;
        }
        unsafe { LLVMBasicBlockAsValue(self) }
    }

    /// Remove this block from its containing function and destroy it
    pub fn delete(self) {
        extern "C" {
            fn LLVMDeleteBasicBlock(bb: Block);
        }
        unsafe { LLVMDeleteBasicBlock(self) }
    }

    /// Remove this block from its containing function, but keep it alive
    pub fn detach(self) -> Self {
        extern "C" {
            fn LLVMRemoveBasicBlockFromParent(bb: Block);
        }
        unsafe {
            LLVMRemoveBasicBlockFromParent(self);
        }
        self
    }

    /// Moves this block before `other`
    pub fn move_before(self, other: Block) {
        extern "C" {
            fn LLVMMoveBasicBlockBefore(bb: Block, before: Block);
        }
        unsafe { LLVMMoveBasicBlockBefore(self, other) }
    }

    /// Moves this block after `other`
    pub fn move_after(self, other: Block) {
        extern "C" {
            fn LLVMMoveBasicBlockAfter(bb: Block, after: Block);
        }
        unsafe { LLVMMoveBasicBlockAfter(self, other) }
    }

    /// Returns the first instruction in this block
    pub fn first(self) -> InstructionBase {
        extern "C" {
            fn LLVMGetFirstInstruction(bb: Block) -> InstructionBase;
        }
        let block = unsafe { LLVMGetFirstInstruction(self) };
        assert!(!block.is_null());
        block
    }

    /// Returns the last instruction in this block
    pub fn last(self) -> InstructionBase {
        extern "C" {
            fn LLVMGetLastInstruction(bb: Block) -> InstructionBase;
        }
        let block = unsafe { LLVMGetLastInstruction(self) };
        assert!(!block.is_null());
        block
    }

    /// Returns this block's terminating instruction, if it contains one
    pub fn terminator(self) -> Option<InstructionBase> {
        extern "C" {
            fn LLVMGetBasicBlockTerminator(bb: Block) -> InstructionBase;
        }
        let value = unsafe { LLVMGetBasicBlockTerminator(self) };
        if value.is_null() {
            None
        } else {
            Some(value)
        }
    }

    /// Returns an iterator over the instructions in this block, starting with the first
    pub fn insts(self) -> impl Iterator<Item = InstructionBase> {
        InstIter::new(self.first())
    }
}

/// Represents a block as an llvm::Value
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct BlockValue(ValueBase);
impl Value for BlockValue {
    fn base(&self) -> ValueBase {
        self.0
    }
}
impl BlockValue {
    pub fn block(self) -> Block {
        extern "C" {
            fn LLVMValueAsBasicBlock(value: BlockValue) -> Block;
        }
        unsafe { LLVMValueAsBasicBlock(self) }
    }
}
impl TryFrom<ValueBase> for BlockValue {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::BasicBlock => Ok(Self(value)),
            _ => Err(InvalidTypeCastError),
        }
    }
}
impl Into<ValueBase> for BlockValue {
    fn into(self) -> ValueBase {
        self.0
    }
}

/// Iterator for blocks in a function
pub(super) struct BlockIter(Block);
impl BlockIter {
    pub(super) fn new(fun: Function) -> Self {
        extern "C" {
            fn LLVMGetFirstBasicBlock(fun: Function) -> Block;
        }
        Self(unsafe { LLVMGetFirstBasicBlock(fun) })
    }
}
impl Iterator for BlockIter {
    type Item = Block;

    fn next(&mut self) -> Option<Self::Item> {
        extern "C" {
            fn LLVMGetNextBasicBlock(bb: Block) -> Block;
        }
        if self.0.is_null() {
            return None;
        }
        let next = self.0;
        self.0 = unsafe { LLVMGetNextBasicBlock(next) };
        Some(next)
    }
}
impl std::iter::FusedIterator for BlockIter {}
