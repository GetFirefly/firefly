use core::cmp;
use core::convert::TryFrom;
use core::mem;
use core::ptr::{self, NonNull};

use alloc::fmt::{self, Debug, Formatter};

use intrusive_collections::container_of;
use intrusive_collections::RBTreeLink;

use liblumen_core::alloc::utils as alloc_utils;
use liblumen_core::alloc::{AllocErr, Layout};

use crate::sorted::{SortKey, SortOrder, Sortable};

use super::{Block, BlockFooter, BlockRef, FreeBlockRef, FreeBlocks};

macro_rules! unlikely {
    ($e:expr) => {{
        #[allow(unused_unsafe)]
        unsafe {
            core::intrinsics::unlikely($e)
        }
    }};
}

/// This struct extends `Block` with extra metadata when free.
///
/// When a block is allocated, it is unlinked from the trees it
/// is a member of, and the memory occupied by the link fields
/// become part of the user data region of the block
pub struct FreeBlock {
    header: Block,
    // Used for an intrusive link for user-provided orderings
    pub(crate) user_link: RBTreeLink,
    // Used for an intrusive link for address-ordered collections
    pub(crate) addr_link: RBTreeLink,
}

impl Debug for FreeBlock {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("FreeBlock")
            .field("header", &self.header)
            .field("user_link", &self.user_link)
            .field("addr_link", &self.addr_link)
            .finish()
    }
}

impl FreeBlock {
    /// Create a new FreeBlock of the given size.
    ///
    /// The block will be marked free but will be unlinked.
    #[inline]
    pub fn new(size: usize) -> Self {
        assert!(size >= Self::min_block_size());
        let mut header = Block::new(size);
        header.set_free();
        Self {
            header,
            user_link: RBTreeLink::default(),
            addr_link: RBTreeLink::default(),
        }
    }

    /// Create a new `FreeBlock` from the given `Block`
    #[inline(always)]
    pub(crate) fn from_block(block: Block) -> Self {
        Self {
            header: block,
            user_link: RBTreeLink::default(),
            addr_link: RBTreeLink::default(),
        }
    }

    /// Gets the usable size of this block's data region
    #[inline(always)]
    pub fn usable_size(&self) -> usize {
        self.header.usable_size()
    }

    /// Updates the usable size metadata for this block
    ///
    /// NOTE: This does not actually change the amount of
    /// memory available to this block
    #[inline(always)]
    fn set_size(&mut self, new_size: usize) {
        self.header.set_size(new_size);
    }

    /// Determines if this block is the last block
    #[inline(always)]
    pub fn is_last(&self) -> bool {
        self.header.is_last()
    }

    /// Marks this block as the last block
    #[inline(always)]
    pub fn set_last(&mut self) {
        self.header.set_last();
    }

    /// Clears the flag marking this block as the last block
    #[inline(always)]
    pub fn clear_last(&mut self) {
        self.header.clear_last();
    }

    // Used in assertions/tests to make sure the header flags
    // match the semantics of this type
    #[cfg(test)]
    #[inline(always)]
    pub(crate) fn is_free(&self) -> bool {
        self.header.is_free()
    }

    /// Returns a `BlockRef` to the next neighboring block, if one exists
    ///
    /// Returns `None` if this block is the last block
    #[inline(always)]
    pub fn next(&self) -> Option<BlockRef> {
        self.header.next()
    }

    /// Returns a `FreeBlockRef` to the previous free block, if possible
    ///
    /// Returns `None` if the previous block is allocated
    ///
    /// NOTE: This relies on the presence of the previous free flag being
    /// set to indicate that a block comes before this block. If the flag
    /// is set, then we know that a `BlockFooter` comes right before this
    /// block's header
    #[inline(always)]
    pub fn prev(&self) -> Option<FreeBlockRef> {
        self.header.prev()
    }

    /// The minimum usable size for a block
    #[inline(always)]
    pub fn min_block_size() -> usize {
        mem::size_of::<FreeBlock>() +
        mem::size_of::<usize>() +
        mem::size_of::<BlockFooter>() -
        // We subtract the size of Block, since Block
        // is always present, and factored in else where
        // The minimum size we actually care about is the
        // minimum _usable_ size
        mem::size_of::<Block>()
    }

    /// This function tries to allocate this block to fulfill the request
    /// represented by `layout`.
    ///
    /// As long as the requested layout fits within the usable space of
    /// this block, including any padding for alignment, then the allocation will
    /// succeed. Otherwise, `Err` is returned.
    ///
    /// The pointer returned is a pointer to the data region of this block, i.e.
    /// it is immediately usable upstream for use by the mutator.
    ///
    /// It is crucial that the caller remove this block from any intrusive collections
    /// it belongs to, _before_ using the block, otherwise traversals of those collections
    /// will reference invalid memory trying to follow links that have been overwritten.
    ///
    /// NOTE: Even though this block makes use of unsafe internal functions, it
    /// is safe to use, because it is not possible for races to occur due to the
    /// lock required at the allocator level.
    ///
    /// NOTE: If the allocator changes such that blocks can be accessed by more
    /// than one thread, the `Block` internals will need to be refactored to handle
    /// that, it is _only_ designed to be accessed by one thread at a time.
    pub fn try_alloc(&mut self, layout: &Layout) -> Result<NonNull<u8>, AllocErr> {
        // This is here as a safety against trying to use a FreeBlock twice
        if unlikely!(!self.header.is_free()) {
            debug_assert!(
                !self.header.is_free(),
                "tried to allocate a free block twice"
            );
            return Err(AllocErr);
        }

        let mut ptr = unsafe { self.header.data() as *mut u8 };

        // We have to make sure the minimum size is big enough to contain block metadata
        let size = cmp::max(layout.size(), Self::min_block_size());

        // Check alignment
        let align = layout.align();
        if !alloc_utils::is_aligned_at(ptr, align) {
            // Need to round up to nearest aligned address
            let aligned_ptr = alloc_utils::align_up_to(ptr, align) as *mut u8;
            assert_ne!(aligned_ptr, ptr);
            // Check size with padding added
            let padding = (aligned_ptr as usize) - (ptr as usize);
            if self.usable_size() < size + padding {
                // No good
                return Err(AllocErr);
            }
            ptr = aligned_ptr
        } else {
            // Alignment is good, check size
            if self.usable_size() < size {
                // No good
                return Err(AllocErr);
            }
        }

        self.header.set_allocated();

        Ok(unsafe { NonNull::new_unchecked(ptr) })
    }

    /// This function is used during allocation to split the current block
    /// if the requested allocation size is significantly smaller than the
    /// amount of usable space in the block. In these cases we can benefit
    /// by splitting out the wasted space into a new free block, leaving it
    /// available for other allocations.
    ///
    /// If unable to split because this block is too small to split, `None` is returned.
    /// Otherwise, the new free block is returned as a mutable reference.
    ///
    /// NOTE: It is expected that the current block will be promptly allocated
    /// and so no attempt is made to create a new block footer for the "old" block.
    /// We do however write both a header and footer for the newly split block.
    pub fn try_split(&mut self, layout: &Layout) -> Option<FreeBlockRef> {
        let mut size = cmp::max(layout.size(), Self::min_block_size());
        // Ensure we have the "real" size of this block
        // This should have been validated by `try_alloc` already
        let align = layout.align();
        let ptr = unsafe { self.header.data() as *mut u8 };
        if !alloc_utils::is_aligned_at(ptr, align) {
            let aligned_ptr = alloc_utils::align_up_to(ptr, align) as *mut u8;
            let padding = (aligned_ptr as usize) - (ptr as usize);
            size = size + padding;
        }

        let usable = self.usable_size();
        let oversized = usable > size;
        // We only split if the resulting split is at least able to hold the
        // minimum allocation size of 8 bytes, including space for the header
        if !oversized {
            return None;
        }
        let split_size = usable - size;
        let min_size = Self::min_block_size();
        let can_split = split_size > min_size;
        if !can_split {
            return None;
        }

        // If we make it here, we can perform the split

        // First, create the new free block header
        let usable_split_size = split_size - mem::size_of::<Block>();
        let mut split_header = Block::new(usable_split_size);
        split_header.set_free();
        // If this block was previously the last block, the split now becomes the last block
        if self.is_last() {
            split_header.set_last();
            self.clear_last();
        }
        // Update the usable size of this block
        self.set_size(size);
        // Write block header for split
        let block_ptr = self as *const _ as *mut u8;
        // `size` here is the usable size, so we need to add on the size of the Block header
        let split_offset = mem::size_of::<Block>() + size;
        let split_ptr = unsafe { block_ptr.add(split_offset) };
        unsafe {
            ptr::write(split_ptr as *mut FreeBlock, split_header.into());
        }
        // Write block footer for split
        // The offset here will be from the base of the
        let split_footer_offset = usable_split_size - mem::size_of::<BlockFooter>();
        let split_footer_ptr = unsafe { split_ptr.add(split_footer_offset) };
        let split_footer = BlockFooter::new(usable_split_size);
        unsafe { ptr::write(split_footer_ptr as *mut BlockFooter, split_footer) };
        // Return split block
        Some(unsafe { FreeBlockRef::from_raw(split_ptr as *mut FreeBlock) })
    }

    /// Combines the referenced free block with its neighboring free blocks
    ///
    /// Returns a `FreeBlockRef` representing the coalesced free block,
    /// if no coalescing operations were needed, the returned ref will
    /// be the same as the input ref
    pub fn coalesce(mut block: FreeBlockRef, free_blocks: &mut FreeBlocks) -> FreeBlockRef {
        if let Some(prev) = block.prev() {
            // Remove this block from the free block tree
            // pending the result of the coalesce operation.
            // It will be added back in the later half of this function
            unsafe {
                free_blocks.remove(prev);
            }
            // Move backwards to beginning of contiguous free block range,
            // then start coalesce operation from there
            return Self::coalesce(prev, free_blocks);
        }

        // We're at the beginning of the range,
        // so extend this block into neighboring blocks
        // one at a time until we reach the next allocated block
        // or the end of the region
        let mut extend_by = 0;
        let mut next_result = block.next();
        let mut is_last = block.is_last();
        loop {
            // No more blocks to walk
            if next_result.is_none() {
                break;
            }

            // Unwrap the result and check whether we've reached the end of the free range
            let mut next_ref = next_result.unwrap();
            let next = next_ref.as_mut();
            if !next.is_free() {
                // This block is free, so make sure the next block knows it
                next.set_prev_free();
                // We're done gathering free blocks to coalesce
                break;
            }

            // If we're becoming the last block, make sure the flag is set
            is_last = next.is_last();
            // The next block is a free block, so get a ref accordingly
            let free = FreeBlockRef::try_from(next_ref).unwrap();
            // Untrack the free block so that it can be combined with this one
            unsafe { free_blocks.remove(free) };
            // Add the size of the free block we're coalescing with
            extend_by += mem::size_of::<Block>() + free.usable_size();
            // Try to move to the next block for the next iteration
            next_result = free.next();
        }

        // Skip modifications if there is nothing to do
        if extend_by > 0 {
            let usable = block.usable_size() + extend_by;

            // Update this block
            block.set_size(usable);
            if is_last {
                block.set_last();
            } else {
                block.clear_last();
            }
            block.write_free_pattern();

            // Write new footer for the coalesced block
            let offset = mem::size_of::<Block>() + usable - mem::size_of::<BlockFooter>();
            unsafe {
                let block_ptr = block.as_ptr() as *const u8;
                let footer_ptr = block_ptr.add(offset);
                ptr::write(footer_ptr as *mut BlockFooter, BlockFooter::new(usable));
            }

            // Add the coalesced block to the free tree
            unsafe { free_blocks.insert(block) };
        }

        block
    }

    #[inline(always)]
    fn write_free_pattern(&mut self) {
        self.header.write_free_pattern()
    }
}

impl From<Block> for FreeBlock {
    fn from(block: Block) -> FreeBlock {
        assert!(block.is_free());
        FreeBlock {
            header: block,
            user_link: RBTreeLink::default(),
            addr_link: RBTreeLink::default(),
        }
    }
}

impl Sortable for FreeBlock {
    type Link = RBTreeLink;

    fn get_value(link: *const Self::Link, order: SortOrder) -> *const Self {
        match order {
            SortOrder::AddressOrder => unsafe { container_of!(link, Self, addr_link) },
            SortOrder::SizeAddressOrder => unsafe { container_of!(link, Self, user_link) },
        }
    }

    fn get_link(value: *const Self, order: SortOrder) -> *const Self::Link {
        match order {
            SortOrder::AddressOrder => unsafe { &(*value).addr_link as *const Self::Link },
            SortOrder::SizeAddressOrder => unsafe { &(*value).user_link as *const Self::Link },
        }
    }

    fn sort_key(&self, order: SortOrder) -> SortKey {
        SortKey::new(order, self.usable_size(), self as *const _ as usize)
    }
}
