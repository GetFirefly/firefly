use std::ffi::c_void;
use std::fmt::{self, Display};
use std::iter::FusedIterator;
use std::ops::Deref;

use super::*;
use crate::support::{self, MlirStringCallback};

extern "C" {
    type MlirBlock;
}

/// Represents a Block whose ownership is managed in Rust
#[repr(transparent)]
pub struct OwnedBlock(Block);
impl Default for OwnedBlock {
    #[inline]
    fn default() -> Self {
        unsafe { mlir_block_create(0, std::ptr::null(), std::ptr::null()) }
    }
}
impl OwnedBlock {
    /// Creates a new block owned by the caller with the given block argument types.
    ///
    /// The `locs` slice should be the same length as `args` and provide the location
    /// for each block argument. If this invariant is violated, the function will panic.
    pub fn new(args: &[TypeBase], locs: &[Location]) -> Self {
        assert_eq!(
            args.len(),
            locs.len(),
            "block argument types and locations must be the same length"
        );
        unsafe { mlir_block_create(args.len(), args.as_ptr(), locs.as_ptr()) }
    }

    /// Releases ownership of this block and returns it as a borrowed block
    pub fn release(self) -> Block {
        let block = self.0;
        std::mem::forget(self);
        block
    }

    #[inline(always)]
    pub fn base(&self) -> Block {
        self.0
    }
}
impl Drop for OwnedBlock {
    fn drop(&mut self) {
        unsafe { mlir_block_destroy(self.0) }
    }
}
impl Deref for OwnedBlock {
    type Target = Block;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl fmt::Pointer for OwnedBlock {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", self.0)
    }
}
impl Eq for OwnedBlock {}
impl PartialEq for OwnedBlock {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl PartialEq<Block> for OwnedBlock {
    fn eq(&self, other: &Block) -> bool {
        self.0.eq(other)
    }
}

/// Represents a borrowed reference to an MLIR block
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Block(*mut MlirBlock);
impl Default for Block {
    fn default() -> Self {
        Self(unsafe { std::mem::transmute::<*mut (), *mut MlirBlock>(::core::ptr::null_mut()) })
    }
}
impl Block {
    /// Returns the next block in this block's region
    #[inline]
    pub fn next(self) -> Option<Block> {
        let block = unsafe { mlir_block_get_next_in_region(self) };
        if block.is_null() {
            None
        } else {
            Some(block)
        }
    }

    /// Returns the containing region, if this block is attached to one
    pub fn region(self) -> Option<Region> {
        let region = unsafe { mlir_block_get_parent_region(self) };
        if region.is_null() {
            None
        } else {
            Some(region)
        }
    }

    /// Returns the containing operation, if this block is attached to an one
    pub fn operation(self) -> Option<OperationBase> {
        let op = unsafe { mlir_block_get_parent_operation(self) };
        if op.is_null() {
            None
        } else {
            Some(op)
        }
    }

    /// Returns the first operation in the block
    #[inline]
    pub fn first(self) -> Option<OperationBase> {
        let op = unsafe { mlir_block_get_first_operation(self) };
        if op.is_null() {
            None
        } else {
            Some(op)
        }
    }

    /// Returns the terminator operation in the block.
    ///
    /// Returns None if there is no terminator.
    #[inline]
    pub fn terminator(self) -> Option<OperationBase> {
        let op = unsafe { mlir_block_get_terminator(self) };
        if op.is_null() {
            None
        } else {
            Some(op)
        }
    }

    /// Appends the given operation to the block
    #[inline]
    pub fn append(self, op: OwnedOperation) {
        unsafe { mlir_block_append_owned_operation(self, op.release()) }
    }

    /// Inserts the given operation at the `index`-th position in the block
    ///
    /// NOTE: This is an expensive operation that scans the block, prefer the
    /// use of `insert_before`/`insert_after` instead.
    #[inline]
    pub fn insert(self, index: usize, op: OwnedOperation) {
        unsafe { mlir_block_insert_owned_operation(self, index, op.release()) }
    }

    /// Inserts `op` before the `before` block. If the `before` block is null,
    /// it prepends the operation to the block.
    #[inline]
    pub fn insert_before(self, before: OperationBase, op: OwnedOperation) {
        unsafe { mlir_block_insert_owned_operation_before(self, before, op.release()) }
    }

    /// Inserts `op` after the `after` block. If the `after` block is null,
    /// it appends the operation to the block.
    #[inline]
    pub fn insert_after(self, after: OperationBase, op: OwnedOperation) {
        unsafe { mlir_block_insert_owned_operation_after(self, after, op.release()) }
    }

    /// Splits this block into two parts, starting before the given operation.
    ///
    /// All of the operations before the given op stay as part of the original block,
    /// and the given op and all ops that follow it are moved to the new block, including
    /// the old terminator.
    ///
    /// NOTE: The original block is left without a terminator.
    #[inline]
    pub fn split(self, before: OperationBase) -> Block {
        unsafe { mlir_block_split_before(self, before) }
    }

    /// Returns the number of arguments of the block
    #[inline]
    pub fn num_arguments(self) -> usize {
        unsafe { mlir_block_get_num_arguments(self) }
    }

    /// Gets the `index`-th argument of the block
    ///
    /// NOTE: This function will panic if the index is out of bounds
    #[inline]
    pub fn get_argument(self, index: usize) -> BlockArgument {
        let arg = unsafe { mlir_block_get_argument(self, index) };
        assert!(
            !arg.is_null(),
            "invalid argument index {}, out of bounds",
            index
        );
        arg
    }

    /// Appends a new block argument of the given type, using the provided source location
    pub fn add_argument(self, ty: TypeBase, loc: Location) -> BlockArgument {
        unsafe { mlir_block_add_argument(self, ty, loc) }
    }

    /// Returns an iterator over the block's arguments
    #[inline]
    pub fn arguments(self) -> impl Iterator<Item = BlockArgument> {
        let num_arguments = self.num_arguments();
        BlockArgumentIter {
            block: self,
            num_arguments,
            pos: 0,
        }
    }

    /// Returns an iterator over the block's operations
    #[inline]
    pub fn iter(self) -> impl Iterator<Item = OperationBase> {
        OperationIter::new(self.first())
    }

    #[inline(always)]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
}
impl fmt::Pointer for Block {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", self.0)
    }
}
impl Eq for Block {}
impl PartialEq for Block {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlir_block_equal(*self, *other) }
    }
}
impl Display for Block {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe {
            mlir_block_print(
                *self,
                support::write_to_formatter,
                f as *mut _ as *mut c_void,
            );
        }
        Ok(())
    }
}

struct BlockArgumentIter {
    block: Block,
    num_arguments: usize,
    pos: usize,
}
impl Iterator for BlockArgumentIter {
    type Item = BlockArgument;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos == self.num_arguments {
            return None;
        }
        let result = self.block.get_argument(self.pos);
        self.pos += 1;
        Some(result)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.num_arguments))
    }
}
impl FusedIterator for BlockArgumentIter {}
impl DoubleEndedIterator for BlockArgumentIter {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.pos == 0 {
            return None;
        }
        let result = self.block.get_argument(self.pos);
        self.pos -= 1;
        Some(result)
    }
}

extern "C" {
    #[link_name = "mlirBlockCreate"]
    fn mlir_block_create(
        num_args: usize,
        args: *const TypeBase,
        locs: *const Location,
    ) -> OwnedBlock;
    #[link_name = "mlirBlockDestroy"]
    fn mlir_block_destroy(block: Block);
    #[link_name = "mlirBlockEqual"]
    fn mlir_block_equal(a: Block, b: Block) -> bool;
    #[link_name = "mlirBlockGetParentOperation"]
    fn mlir_block_get_parent_operation(block: Block) -> OperationBase;
    #[link_name = "mlirBlockGetParentRegion"]
    fn mlir_block_get_parent_region(block: Block) -> Region;
    #[link_name = "mlirBlockGetNextInRegion"]
    fn mlir_block_get_next_in_region(block: Block) -> Block;
    #[link_name = "mlirBlockGetFirstOperation"]
    fn mlir_block_get_first_operation(block: Block) -> OperationBase;
    #[link_name = "mlirBlockGetTerminator"]
    fn mlir_block_get_terminator(block: Block) -> OperationBase;
    #[link_name = "mlirBlockAppendOwnedOperation"]
    fn mlir_block_append_owned_operation(block: Block, op: OperationBase);
    #[link_name = "mlirBlockInsertOwnedOperation"]
    fn mlir_block_insert_owned_operation(block: Block, index: usize, op: OperationBase);
    #[link_name = "mlirBlockInsertOwnedOperationAfter"]
    fn mlir_block_insert_owned_operation_after(
        block: Block,
        reference: OperationBase,
        op: OperationBase,
    );
    #[link_name = "mlirBlockInsertOwnedOperationBefore"]
    fn mlir_block_insert_owned_operation_before(
        block: Block,
        reference: OperationBase,
        op: OperationBase,
    );
    #[link_name = "mlirBlockSplitBefore"]
    fn mlir_block_split_before(block: Block, op: OperationBase) -> Block;
    #[link_name = "mlirBlockGetNumArguments"]
    fn mlir_block_get_num_arguments(block: Block) -> usize;
    #[link_name = "mlirBlockAddArgument"]
    fn mlir_block_add_argument(block: Block, ty: TypeBase, loc: Location) -> BlockArgument;
    #[link_name = "mlirBlockGetArgument"]
    fn mlir_block_get_argument(block: Block, index: usize) -> BlockArgument;
    #[link_name = "mlirBlockPrint"]
    fn mlir_block_print(block: Block, callback: MlirStringCallback, userdata: *const c_void);
}
