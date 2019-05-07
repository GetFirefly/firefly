use core::mem;
use core::ptr::{self, NonNull};
use core::alloc::Layout;

use intrusive_collections::container_of;

use crate::block::{Block, FreeBlockTree};
use crate::sorted::{Link, Sortable, SortOrder, SortKey};

/// This struct is the carrier type for large allocations that
/// exceed a given threshold, typically anything larger than
/// `size_classes::MAX_SIZE_CLASS`.
///
/// This type of carrier only contains a single block, and is
/// optimized for that case.
///
/// NOTE: Single-block carriers are currently freed when the
/// block they contain is freed, but it may be that we will want
/// to cache some number of these carriers if large allocations
/// are frequent and reuse known to be likely
#[repr(C)]
#[derive(Debug)]
pub struct SingleBlockCarrier<L: Link> {
    pub(crate) size: usize,
    pub(crate) layout: Layout,
    pub(crate) link: L,
}
impl<L> SingleBlockCarrier<L>
where
    L: Link,
{
    /// Returns the Layout used for the data contained in this carrier
    #[inline]
    pub fn layout(&self) -> Layout {
        self.layout.clone()
    }

    /// Returns a raw pointer to the data region in the sole block of this carrier
    ///
    /// NOTE: You may get the same pointer by asking a Block for its data, but
    /// this function bypasses the need to get the block and then ask for the data.
    /// This is "safe" since the data location is known by the carrier, or at least
    /// is able to be calculated by the carrier.
    ///
    /// NOTE: This is unsafe because 1.) the pointer is unmanaged, and 2.) if the
    /// pointer or a reference constructed from that pointer lives longer than this
    /// carrier, then it will be invalid, potentially allowing use-after-free.
    /// Additionally, the data pointer is the pointer given to `free`, so there is
    /// the potential that multiple pointers could permit double-free scenarios.
    /// This is less of a risk as the current implementation only frees memory if
    /// an allocated carrier can be found which owns the pointer, and after the first
    /// free, that can't happen..
    #[allow(unused)]
    #[inline]
    pub unsafe fn data<T>(&self) -> *const T {
        let (_layout, data_offset) = Layout::new::<Self>()
            .extend(self.layout.clone())
            .unwrap();

        let ptr = self as *const _ as *const u8;
        ptr.offset(data_offset as isize) as *const T
    }

    /// Calculate the usable size of this carrier, specifically the size
    /// of the data region contained in this carrier's block
    #[allow(unused)]
    #[inline]
    pub fn usable_size(&self) -> usize {
        self.size - self.layout.size()
    }

    /// Determines if the given pointer belongs to this carrier, this is
    /// primarily used in `free` to determine which carrier to free.
    #[inline]
    pub fn owns(&self, ptr: *const u8) -> bool {
        let this = self as *const _ as usize;
        let ptr = ptr as usize;

        // Belongs to a lower-addressed carrier
        if ptr < this {
            return false;
        }

        // Belongs to a higher-addressed carrier
        if (this - ptr) > self.size {
            return false;
        }

        // Falls within this carrier's address range
        true
    }
}

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
    // The first block in
    // Used to store the intrusive link to a size + address ordered tree,
    pub(crate) link: L,
    // This field stores an intrusive red/black tree where blocks are tracked
    pub(crate) blocks: FreeBlockTree,
}
impl<L> MultiBlockCarrier<L>
where
    L: Link,
{
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
    unsafe fn head(&self) -> NonNull<Block> {
        let ptr = (self as *const Self).offset(1) as *mut Block;
        NonNull::new_unchecked(ptr)
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
        // Starting at the head of the block list, traverse
        // until we find a block which can satisfy the given request
        let mut block = self.head();
        loop {
            let blk = block.as_mut();
            if let Ok(ptr) = blk.try_alloc(layout) {
                return Some(ptr);
            }

            match blk.next() {
                None => return None,
                Some(next) => {
                    block = next;
                }
            }
        }
    }

    #[inline]
    pub unsafe fn realloc_block(&self, ptr: *mut u8, layout: &Layout, new_size: usize) -> Option<NonNull<u8>> {
        let old_size = layout.size();
        // Locate the current block
        let mut result = Some(self.head());
        loop {
            if result.is_none() {
                break;
            }
            let mut block = result.unwrap();
            let blk = block.as_mut();
            let raw = blk as *const _ as *const u8;
            if raw == ptr {
                if old_size < new_size {
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
        blk.free();
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
            let blk = block.as_mut();
            let raw = blk as *const _ as *const u8;
            if (raw as *const u8) == ptr {
                return blk.free();
            }

            match blk.next() {
                None => return,
                Some(next) => {
                    block = next;
                }
            }
        }
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
        unsafe {  &(*value).link as *const L }
    }

    fn sort_key(&self, order: SortOrder) -> SortKey {
        SortKey::new(order, self.usable_size(), self as *const _ as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_block_carrier_test() {
        unimplemented!()
    }

    #[test]
    fn multi_block_carrier_test() {
        unimplemented!()
    }
}
