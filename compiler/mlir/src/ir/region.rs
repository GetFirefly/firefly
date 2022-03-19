use std::fmt;
use std::iter::FusedIterator;
use std::ops::Deref;

use super::*;

extern "C" {
    type MlirRegion;
}

/// Represents a Region whose ownership is managed in Rust
#[repr(transparent)]
pub struct OwnedRegion(Region);
impl OwnedRegion {
    /// Creates a new Region allocated by MLIR, but owned by the caller
    pub fn new() -> Self {
        unsafe { mlir_region_create() }
    }

    /// Releases ownership of this region and returns it as a borrowed region
    pub fn release(self) -> Region {
        let region = self.0;
        std::mem::forget(self);
        region
    }
}
impl Drop for OwnedRegion {
    fn drop(&mut self) {
        unsafe { mlir_region_destroy(self.0) }
    }
}
impl Deref for OwnedRegion {
    type Target = Region;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl fmt::Pointer for OwnedRegion {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", self.0)
    }
}
impl Eq for OwnedRegion {}
impl PartialEq for OwnedRegion {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl PartialEq<Region> for OwnedRegion {
    fn eq(&self, other: &Region) -> bool {
        self.0.eq(other)
    }
}

/// Represents a container of Blocks within an Operation
///
/// An operation may have zero or more regions, and a region may have zero or more blocks,
/// and blocks may have zero or more operations.
///
/// NOTE: An empty region is more of a special case, but is used in some circumstances,
/// such as functions, to represent that it is a declaration vs a definition.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Region(*mut MlirRegion);
impl Region {
    /// Gets the first block in the region
    pub fn entry(self) -> Option<Block> {
        let block = unsafe { mlir_region_get_first_block(self) };
        if block.is_null() {
            None
        } else {
            Some(block)
        }
    }

    /// Takes a block owned by the caller and appends it to the given region
    pub fn append(self, block: OwnedBlock) {
        unsafe { mlir_region_append_owned_block(self, block.release()) }
    }

    /// Takes a block owned by the caller and inserts it at `index` position
    /// in the region.
    ///
    /// NOTE: This is an expensive operation that scans the region, prefer the
    /// use of `insert_after`/`insert_before` instead.
    pub fn insert(self, index: usize, block: OwnedBlock) {
        unsafe {
            mlir_region_insert_owned_block(self, index, block.release());
        }
    }

    /// Takes a block owned by the caller and inserts it before the reference block
    /// in the given region. The reference block must belong to the region. If the
    /// reference block is null, the block being inserted is prepended to the region.
    pub fn insert_before(self, before: Block, block: OwnedBlock) {
        unsafe {
            mlir_region_insert_owned_block_before(self, before, block.release());
        }
    }

    /// Takes a block owned by the caller and inserts it after the reference block
    /// in the given region. The reference block must belong to the region. If the
    /// reference block is null, the block being inserted is appended to the region.
    pub fn insert_after(self, after: Block, block: OwnedBlock) {
        unsafe {
            mlir_region_insert_owned_block_after(self, after, block.release());
        }
    }

    /// Returns an iterator over the blocks in this region
    pub fn iter(self) -> impl Iterator<Item = Block> {
        RegionIter {
            current: self.entry(),
        }
    }

    #[inline(always)]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
}
impl fmt::Pointer for Region {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", self.0)
    }
}
impl Eq for Region {}
impl PartialEq for Region {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlir_region_equal(*self, *other) }
    }
}

/// Represents an iterator of Blocks within a region
struct RegionIter {
    current: Option<Block>,
}
impl Iterator for RegionIter {
    type Item = Block;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_none() {
            return None;
        }

        let current = self.current.unwrap();
        self.current = current.next();
        Some(current)
    }
}
impl FusedIterator for RegionIter {}

extern "C" {
    #[link_name = "mlirRegionCreate"]
    fn mlir_region_create() -> OwnedRegion;
    #[link_name = "mlirRegionDestroy"]
    fn mlir_region_destroy(region: Region);
    #[link_name = "mlirRegionEqual"]
    fn mlir_region_equal(a: Region, b: Region) -> bool;
    #[link_name = "mlirRegionGetFirstBlock"]
    fn mlir_region_get_first_block(region: Region) -> Block;
    #[link_name = "mlirRegionAppendOwnedBlock"]
    fn mlir_region_append_owned_block(region: Region, block: Block);
    #[link_name = "mlirRegionInsertOwnedBlock"]
    fn mlir_region_insert_owned_block(region: Region, index: usize, block: Block);
    #[link_name = "mlirRegionInsertOwnedBlockAfter"]
    fn mlir_region_insert_owned_block_after(region: Region, reference: Block, block: Block);
    #[link_name = "mlirRegionInsertOwnedBlockBefore"]
    fn mlir_region_insert_owned_block_before(region: Region, reference: Block, block: Block);
}
