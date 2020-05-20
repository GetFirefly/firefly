use core::cell::RefCell;
use core::mem;
use core::ptr::{self, NonNull};

use intrusive_collections::container_of;

use liblumen_core::alloc::Layout;

use crate::blocks::{Block, BlockRef, FreeBlock, FreeBlockRef, FreeBlocks};
use crate::sorted::{Link, SortKey, SortOrder, Sortable};

/// This struct represents a carrier type which can contain
/// multiple blocks of variable size, and is designed specifically
/// for that case. For a carrier optimized for fixed size allocations,
/// see the documentation for `SlabCarrier`.
///
/// This type of multi-block carrier carries an intrusive red/black
/// tree for tracking free blocks available for use, making best fit
/// searches in `O(log N)`.
///
/// It also contains an intrusive link for use by a parent allocator
/// which wants to store carriers in a collection for optimal searches.
///
/// NOTE: This carrier type is designed to be created once and reused
/// indefinitely. While they can be freed when all blocks are free, the
/// current set of allocators do not ever free these carriers once allocated.
/// That will need to change eventually, but is not a high-priority issue
/// for now.
///
/// TODO: Support carrier migration
#[repr(C)]
pub struct MultiBlockCarrier<L: Link> {
    // The total size of this carrier
    pub(crate) size: usize,
    // Used to store the intrusive link to a size + address ordered tree,
    pub(crate) link: L,
    // This field stores an intrusive red/black tree where blocks are tracked
    pub(crate) blocks: RefCell<FreeBlocks>,
}
impl<L> MultiBlockCarrier<L>
where
    L: Link,
{
    #[inline]
    pub unsafe fn init(ptr: NonNull<u8>, size: usize) -> *mut Self {
        // Write carrier header to given memory region
        let carrier = ptr.as_ptr() as *mut MultiBlockCarrier<L>;
        ptr::write(
            carrier,
            MultiBlockCarrier {
                size,
                link: L::default(),
                blocks: RefCell::new(FreeBlocks::new(SortOrder::SizeAddressOrder)),
            },
        );
        // Get a mutable reference for later
        let this = &mut *carrier;
        // Write initial free block header
        let block = carrier.add(1) as *mut FreeBlock;
        let usable = size - mem::size_of::<Block>() - mem::size_of::<MultiBlockCarrier<L>>();
        let mut free_block = FreeBlock::new(usable);
        free_block.set_last();
        ptr::write(block, free_block);

        // Add free block to internal free list
        let mut blocks = this.blocks.borrow_mut();
        blocks.insert(FreeBlockRef::from_raw(block));

        carrier
    }

    /// Calculates the usable size of this carrier, specifically the
    /// size available to be allocated to blocks. In practice, the
    /// usable size for user allocations is smaller, as block headers
    /// take up some space in the carrier
    #[inline]
    pub fn usable_size(&self) -> usize {
        self.size - mem::size_of::<Self>()
    }

    /// Gets a reference to the first block in this carrier.
    /// There is always at least one block, so there is no risk
    /// of this returning an invalid reference.
    ///
    /// NOTE: This is unsafe because a reference that outlives this
    /// carrier will become invalid, potentially allowing use-after-free.
    #[inline]
    fn head(&self) -> BlockRef {
        unsafe {
            let ptr = (self as *const Self).add(1) as *mut Block;
            BlockRef::from_raw(ptr)
        }
    }

    /// Tries to satisfy an allocation request using a block in this carrier.
    /// If successful, returns a raw pointer to the data region of that block.
    ///
    /// NOTE: This is unsafe because a raw pointer is returned, it is important
    /// that these pointers are not allowed to live beyond the life of both the
    /// block that owns them, and the carrier itself. If a pointer is double-freed
    /// then other references will be invalidated, resulting in undefined behavior,
    /// the worst of which is silent corruption of memory due to reuse of blocks.
    ///
    /// Futhermore, if the carrier itself is freed when there are still pointers
    /// to blocks in the carrier, the same undefined behavior is possible, though
    /// depending on how the underyling memory is allocated, it may actually produce
    /// SIGSEGV or equivalent.
    #[inline]
    pub unsafe fn alloc_block(&self, layout: &Layout) -> Option<NonNull<u8>> {
        // Try to find a block that will fit
        let mut blocks = self.blocks.borrow_mut();
        let result = blocks.find_best_fit(layout);
        // No fit, then we're done
        if result.is_none() {
            return None;
        }
        // We have a fit, so allocate the block and update relevant metadata
        let mut allocated = result.unwrap();
        let ptr = allocated
            .try_alloc(layout)
            .expect("find_best_fit and try_alloc disagreed!");
        blocks.remove(allocated);
        // Allocate this block
        // Check if we should split the block first
        if let Some(split_block) = allocated.try_split(layout) {
            // Add the newly split block to the free blocks tree
            blocks.insert(split_block);
            // We're done, return the userdata pointer
            return Some(ptr);
        }
        // There was no split, so check if the neighboring block
        // thinks we're free and fix that
        if let Some(mut neighbor) = allocated.next() {
            neighbor.as_mut().set_prev_allocated();
        }
        // Return the userdata pointer
        Some(ptr)
    }

    #[inline]
    pub unsafe fn realloc_block(
        &self,
        ptr: *mut u8,
        layout: &Layout,
        new_size: usize,
    ) -> Option<NonNull<u8>> {
        // Locate the given block
        // The pointer given is for the aligned data region, so we need
        // to find the block which contains this pointer
        let old_size = layout.size();
        let mut result = Some(self.head());
        loop {
            if result.is_none() {
                break;
            }
            let mut block = result.unwrap();
            let blk = block.as_mut();
            if blk.owns(ptr) {
                if old_size <= new_size {
                    // Try to grow in place, otherwise proceed to realloc
                    if blk.grow_in_place(new_size) {
                        return Some(NonNull::new_unchecked(ptr));
                    } else {
                        break;
                    }
                } else {
                    // Shrink in place, this always succeeds for now
                    blk.shrink_in_place(new_size);
                    return Some(NonNull::new_unchecked(ptr));
                }
            }

            result = blk.next();
        }

        // If current is None, this realloc call was given with an invalid pointer
        assert!(result.is_some(), "possible use-after-free");
        let mut block = result.unwrap();
        let blk = block.as_mut();

        // Unable to alloc in previous block, so this requires a new allocation
        let layout = Layout::from_size_align_unchecked(new_size, layout.align());
        let new_block = self.alloc_block(&layout)?;
        let new_ptr = new_block.as_ptr();
        // Copy old data into new block
        ptr::copy_nonoverlapping(ptr, new_ptr, old_size);
        // Free old block
        let free_block = blk.free();
        let mut blocks = self.blocks.borrow_mut();
        blocks.insert(free_block);
        // Return new block
        Some(new_block)
    }

    /// Frees a block in this carrier.
    ///
    /// The memory backing the block is not actually released to the operating system,
    /// instead the block is marked free and made available for new allocation requests.
    ///
    /// NOTE: This is unsafe:
    ///
    /// - It is critical to ensure frees occur when only one pointer/reference exists to the block,
    ///   otherwise it is possible to double-free or corrupt new allocations in that block
    /// - Since blocks are reused, it is imperative that no pointers/references refer to the data
    ///   region of the freed block after this function is called, or that memory can be corrupted,
    ///   or at a minimum result in undefined behavior.
    #[inline]
    pub unsafe fn free_block(&self, ptr: *const u8, _layout: Layout) {
        // The pointer is for the start of the aligned data region
        // Locate the block indicated by the pointer
        let mut block = self.head();
        loop {
            if block.owns(ptr) {
                let blk = block.as_mut();
                // Free the block
                let mut blocks = self.blocks.borrow_mut();
                let freed = blk.free();
                // We don't add `freed` to the free blocks tree yet,
                // we rely on the coalesce operation to combine free
                // blocks first, and then the resulting block is added
                // to the tree here
                let _coalesced = FreeBlock::coalesce(freed, &mut blocks);
                // Done
                return;
            }

            match block.next() {
                None => return,
                Some(next) => {
                    block = next;
                }
            }
        }
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn num_blocks_free(&self) -> usize {
        self.blocks.borrow().count()
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn num_blocks(&self) -> usize {
        let mut block = self.head();
        let mut count = 1;
        loop {
            if let Some(next_block) = block.next() {
                block = next_block;
                count += 1;
                continue;
            }

            break;
        }

        count
    }
}

impl<L> Sortable for MultiBlockCarrier<L>
where
    L: Link,
{
    type Link = L;

    fn get_value(link: *const L, _order: SortOrder) -> *const MultiBlockCarrier<L> {
        // the second `link` is the name of the link field in the carrier struct
        unsafe { container_of!(link, MultiBlockCarrier<L>, link) }
    }

    fn get_link(value: *const Self, _order: SortOrder) -> *const L {
        unsafe { &(*value).link as *const L }
    }

    fn sort_key(&self, order: SortOrder) -> SortKey {
        SortKey::new(order, self.usable_size(), self as *const _ as usize)
    }
}

#[cfg(test)]
mod tests {
    use core::alloc::AllocRef;

    use super::*;

    use intrusive_collections::RBTreeLink;
    use liblumen_core::alloc::{prelude::*, SysAlloc};

    use crate::blocks::FreeBlockRef;
    use crate::carriers::SUPERALIGNED_CARRIER_SIZE;

    #[test]
    fn multi_block_carrier_test() {
        // Use super-aligned size from std_alloc
        let size = SUPERALIGNED_CARRIER_SIZE;
        let carrier_layout = Layout::from_size_align(size, size).unwrap();
        // Allocate region
        let alloc_block = SysAlloc::get_mut()
            .alloc(carrier_layout, AllocInit::Uninitialized)
            .unwrap();
        // Get pointer to carrier header location
        let carrier = alloc_block.ptr.as_ptr() as *mut MultiBlockCarrier<RBTreeLink>;
        // Write initial carrier header
        unsafe {
            ptr::write(
                carrier,
                MultiBlockCarrier {
                    size,
                    link: RBTreeLink::default(),
                    blocks: RefCell::new(FreeBlocks::new(SortOrder::SizeAddressOrder)),
                },
            );
        }
        let mbc = unsafe { &mut *carrier };
        // Write initial free block
        let usable =
            size - mem::size_of::<Block>() - mem::size_of::<MultiBlockCarrier<RBTreeLink>>();
        let block = unsafe { carrier.add(1) as *const _ as *mut FreeBlock };
        unsafe {
            let mut header = Block::new(usable);
            header.set_free();
            header.set_last();
            ptr::write(block, FreeBlock::from(header));
            let mut blocks = mbc.blocks.borrow_mut();
            blocks.insert(FreeBlockRef::from_raw(block));
        }
        assert_eq!(mbc.num_blocks_free(), 1);
        assert_eq!(mbc.num_blocks(), 1);
        // Allocate 4k large, word-aligned block using newly allocated carrier
        // This should result in a split, and an extra block added
        let layout = Layout::from_size_align(4096, 8).unwrap();
        let block = unsafe { mbc.alloc_block(&layout) };
        assert!(block.is_some());
        assert_eq!(mbc.num_blocks_free(), 1);
        assert_eq!(mbc.num_blocks(), 2);
        // Freeing the allocated block will coalesce these blocks into one again
        let block_ref = block.unwrap();
        unsafe {
            mbc.free_block(block_ref.as_ptr(), layout);
        }
        assert_eq!(mbc.num_blocks_free(), 1);
        assert_eq!(mbc.num_blocks(), 1);
        // Cleanup
        drop(mbc);
        unsafe { SysAlloc::get_mut().dealloc(alloc_block.ptr, carrier_layout) };
    }
}
