#![allow(unused)]
use core::mem;
use core::ptr::{self, NonNull};
use core::alloc::{Layout, AllocErr};
use core::default::Default;
use core::intrinsics::unlikely;

use intrusive_collections::container_of;
use intrusive_collections::{UnsafeRef, RBTree, RBTreeLink};

use crate::utils;
use crate::sorted::Link;
use crate::sorted::{Sortable, SortOrder, SortKey};
use crate::sorted::SortedKeyAdapter;

/// This struct is used to represent the header of a block,
/// at a minimum it contains both the block size, and three
/// flags: whether the block is free, whether its previous neighbor
/// is free (for block coalescing), and whether it is the last block
/// in its carrier.
///
/// This header/struct is shared in both allocated and free blocks,
/// but allocated blocks _only_ have this header, while free blocks
/// are extended with additional metadata, primarily the intrusive
/// links for the collections that maintain free block information.
/// See `FreeBlock` for more details on those.
///
/// NOTE: This struct is `repr(transparent)` because we want it to
/// share the same in memory representation as `usize`, as well as
/// share the same ABI semantics as `usize` in calls. On some platforms
/// arguments/return values of a scalar type are treated differently
/// than structs containing a single field of the same type. This is
/// of primary importance with FFI, but while we may not be using `Block`
/// across FFI boundaries, we choose to use an FFI-safe representation
/// for safety, and to indicate to Rust that the types are equivalent.
///
/// See [the Unstable Book](https://doc.rust-lang.org/1.26.2/unstable-book/language-features/repr-transparent.html)
/// for more details the `repr(transparent)` attribute.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct Block(usize);
impl Default for Block {
    fn default() -> Self {
        Block(0)
    }
}

/// This struct extends `Block` with extra metadata when free,
/// and uses a packed representation to ensure that we do not
/// waste any bytes to padding, as the fields are infrequently
/// used, so even if unaligned accesses are possible and penalized
/// (not typically an issue on modern systems), it is worth the tradeoff.
///
/// When a block is allocated, it is unlinked from the trees it
/// is a member of, and the memory occupied by the link fields
/// become part of the user data region of the block
#[derive(Default)]
#[repr(packed)]
pub struct FreeBlock<L: Link> {
    header: Block,
    // Used for an intrusive link for user-provided orderings
    user_link: L,
    // Used for an intrusive link for address-ordered collections
    addr_link: L,
}
impl<L: Link> FreeBlock<L> {
    #[inline]
    pub fn usable_size(&self) -> usize {
        unsafe { self.header.usable_size() }
    }
}
impl<L> Sortable for FreeBlock<L>
where
    L: Link,
{
    type Link = L;

    fn get_value(link: *const Self::Link, order: SortOrder) -> *const Self {
        match order {
            SortOrder::AddressOrder => {
                unsafe { container_of!(link, Self, addr_link) }
            }
            SortOrder::SizeAddressOrder => {
                unsafe { container_of!(link, Self, user_link) }
            }
        }
    }

    fn get_link(value: *const Self, order: SortOrder) -> *const Self::Link {
        match order {
            SortOrder::AddressOrder =>
                unsafe {  &(*value).addr_link as *const Self::Link },
            SortOrder::SizeAddressOrder =>
                unsafe {  &(*value).user_link as *const Self::Link },
        }
    }

    fn sort_key(&self, order: SortOrder) -> SortKey {
        SortKey::new(order, self.usable_size(), self as *const _ as usize)
    }
}

impl Block {
    // This mask is used to strip the high bits from the raw header value.
    // These high bits (four to be precise) are used to store flags
    #[cfg(target_pointer_width = "32")]
    const HIGH_BIT_MASK: usize = 0xF000_0000;
    #[cfg(target_pointer_width = "64")]
    const HIGH_BIT_MASK: usize = 0xF000_0000_0000_0000;
    // Used to shift flags into the high bits
    const FLAG_SHIFT: usize = (mem::size_of::<usize>() * 8) - 4;
    // Marks a block as free
    const FREE_FLAG: usize = 1 << Self::FLAG_SHIFT;
    // Marks a block as the last block in the carrier
    const LAST_FLAG: usize = 1 << (Self::FLAG_SHIFT + 1);
    // Indicates the previous neighboring block is free
    // This is used to drive coalescing of blocks
    const PREV_FREE_FLAG: usize = 1 << (Self::FLAG_SHIFT + 2);
    // The byte pattern used when a block is freed or initialized
    const FREE_PATTERN: u8 = 0x57;

    #[inline]
    pub fn usable_size(&self) -> usize {
        self.0 & !Self::HIGH_BIT_MASK
    }

    #[inline]
    pub unsafe fn data(&self) -> *const u8 {
        let raw = self as *const Block;
        raw.offset(1) as *const u8
    }

    #[inline(always)]
    pub fn is_free(&self) -> bool {
        (self.0 & Self::FREE_FLAG) == Self::FREE_FLAG
    }

    #[inline(always)]
    pub fn set_free(&mut self) {
        self.0 |= Self::FREE_FLAG;
    }

    #[inline(always)]
    pub fn set_allocated(&mut self) {
        self.0 &= !Self::FREE_FLAG;
    }

    #[inline(always)]
    pub fn is_prev_free(&self) -> bool {
        (self.0 & Self::PREV_FREE_FLAG) == Self::PREV_FREE_FLAG
    }

    #[inline(always)]
    pub fn set_prev_free(&mut self) {
        self.0 |= Self::PREV_FREE_FLAG
    }

    #[inline(always)]
    pub fn set_prev_allocated(&mut self) {
        self.0 &= !Self::PREV_FREE_FLAG
    }

    #[inline(always)]
    pub fn is_last(&self) -> bool {
        (self.0 & Self::LAST_FLAG) == Self::LAST_FLAG
    }

    #[inline(always)]
    pub fn set_last(&mut self) {
        self.0 |= Self::LAST_FLAG;
    }

    #[inline(always)]
    pub fn clear_last(&mut self) {
        self.0 &= !Self::LAST_FLAG;
    }

    /// Locates the next block following this block, if it exists.
    ///
    /// TODO: This needs careful testing
    #[inline]
    pub fn next(&self) -> Option<NonNull<Block>> {
        if self.is_last() {
            return None;
        }

        let size = self.usable_size();
        let ptr = unsafe { self.data() };

        Some(unsafe {
            NonNull::new_unchecked(ptr.offset(size as isize) as *mut Block)
        })
    }

    /// Locates the block footer for this block, if it exists.
    ///
    /// NOTE: This function returns an Option because a footer is only
    /// present when the block is free,
    #[inline]
    pub fn footer(&self) -> Option<NonNull<BlockFooter>> {
        if !self.is_free() {
            return None;
        }

        let size = self.usable_size();
        let offset = (size - mem::size_of::<BlockFooter>());
        unsafe {
            let ptr = self.data();
            let footer_ptr = ptr.offset(offset as isize) as *mut BlockFooter;
            Some(NonNull::new_unchecked(footer_ptr))
        }
    }

    /// This function tries to allocate this block to fulfill the request
    /// represented by `layout`. If the block is already allocated, it returns
    /// `Err`. As long as the requested layout fits within the usable space of
    /// this block, including any padding for alignment, then the allocation will
    /// succeed. Otherwise, `Err` is returned.
    ///
    /// The pointer returned is a pointer to the data region of this block, i.e.
    /// it is immediately usable upstream for use by the mutator.
    ///
    /// NOTE: Even though this block makes use of unsafe internal functions, it
    /// is safe to use, because it is not possible for races to occur due to the
    /// lock required at the allocator level.
    ///
    /// NOTE: If the allocator changes such that blocks can be accessed by more
    /// than one thread, the `Block` internals will need to be refactored to handle
    /// that, it is _only_ designed to be accessed by one thread at a time.
    pub fn try_alloc(&mut self, layout: &Layout) -> Result<NonNull<u8>, AllocErr> {
        if unsafe { unlikely(!self.is_free()) } {
            return Err(AllocErr);
        }

        let mut ptr = unsafe { self.data() as *mut u8 };

        // Check alignment
        let align = layout.align();
        if utils::is_aligned_at(ptr, align) {
            // Need to round up to nearest aligned address
            let aligned_ptr = utils::align_up_to(ptr, align) as *mut u8;
            assert_eq!(aligned_ptr, ptr);
            // Check size with padding added
            let padding = (aligned_ptr as usize) - (ptr as usize);
            if self.usable_size() < layout.size() + padding {
                // No good
                return Err(AllocErr);
            }
            ptr = aligned_ptr
        } else {
            // Alignment is good, check size
            if self.usable_size() < layout.size() {
                // No good
                return Err(AllocErr);
            }
        }

        self.set_allocated();

        Ok(unsafe { NonNull::new_unchecked(ptr) })
    }

    /// Attempts to grow the block size in place.
    /// Returns true if successful, false if unable to grow in place.
    ///
    /// NOTE: This currently returns true only if `new_size` fits in the
    /// current usable space of the block, so it is not particularly useful.
    ///
    /// TODO: Permit coalescing with the next block(s) to grow.
    #[inline]
    pub fn grow_in_place(&mut self, new_size: usize) -> bool {
        self.usable_size() >= new_size
    }

    /// Attemps to shrink the block size in place.
    /// Returns true if successful, false if unable to shrink in place.
    ///
    /// NOTE: This currently always returns true if `new_size` fits in
    /// the current usable space of the block.
    ///
    /// TODO: Should shrink by splitting out new free block, then
    /// adjusting size of old block, currently we don't actually
    /// shrink, but it is a placeholder for that functionality
    #[inline]
    pub fn shrink_in_place(&mut self, new_size: usize) -> bool {
        self.usable_size() >= new_size
    }

    /// Free this block
    ///
    /// NOTE: This does not release memory back to the operating system
    ///
    /// NOTE: This function does most of the work necessary to initialize
    /// the freed block, but it *does not* initialize the links in `FreeBlock<T>`,
    /// since we don't know the type of the link here. Instead it is up to
    /// the carrier to initialize the links as part of the work it performs during
    /// deallocation of a block.
    #[inline]
    pub fn free(&mut self) {
        // Freeing a block here only means marking this block as free,
        // the higher level carrier code is responsible for coalescing
        // blocks and releasing memory as appropriate

        // Set free flag
        self.set_free();
        // If not last, update the neighboring block to indicate that
        // this block is free, so that coalescing operations can join them
        if !self.is_last() {
            let mut next = self.next().unwrap();
            unsafe { next.as_mut().set_prev_free() };
        }
        // Write free pattern over this block, a no-op when not in debug mode
        self.write_free_pattern();
        // Add block footer
        let size = self.usable_size();
        let offset = size - mem::size_of::<BlockFooter>();
        unsafe {
            let ptr = self.data();
            let footer_ptr = ptr.offset(offset as isize) as *mut BlockFooter;
            ptr::write(
                footer_ptr,
                BlockFooter(size),
            );
        }
    }

    /// Convert a `Block` reference into a `FreeBlock<T>` reference.
    ///
    /// Returns `None` if this block is not free.
    #[inline]
    pub fn to_free_block<T: Link>(&self) -> Option<&FreeBlock<T>> {
        if !self.is_free() {
            return None;
        }
        let ptr = self as *const _ as *const FreeBlock<T>;
        Some(unsafe { &*ptr })
    }

    /// Convert a mutable `Block` reference into a mutable `FreeBlock<T>` reference.
    ///
    /// Returns `None` if this block is not free.
    #[inline]
    pub fn to_free_block_mut<T: Link>(&mut self) -> Option<&mut FreeBlock<T>> {
        if !self.is_free() {
            return None;
        }
        let ptr = self as *mut _ as *mut FreeBlock<T>;
        Some(unsafe { &mut *ptr })
    }

    #[cfg(debug_assertions)]
    fn write_free_pattern(&self) {
        // The data pointer must be word aligned
        let ptr = unsafe { self.data() };
        assert_word_aligned!(ptr);

        // Write the pattern over all bytes in the block except those
        // containing the block header itself
        unsafe {
            ptr::write_bytes(
                ptr as *mut u8,
                Self::FREE_PATTERN,
                self.usable_size()
            );
        }
    }

    #[cfg(not(debug_assertions))]
    fn write_free_pattern(&self) {}
}

/// This struct is similar in nature to `Block`, but is used
/// to store the size of the preceding data region when the block is free.
///
/// When the next neighboring block is freed, a check is performed to
/// see if it can be combined with its preceding block, the one containing
/// this footer, this combining operation is called "coalescing".
///
/// If the blocks can be coalesced, then the footer is used to get the size of this
/// block, so that the address of the block header can be calculated. That address
/// is then used to access the header and update it with new metadata
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct BlockFooter(usize);
impl BlockFooter {
    #[inline]
    pub const fn new(size: usize) -> Self {
        Self(size)
    }
}
impl BlockFooter {
    #[inline]
    pub fn usable_size(&self) -> usize {
        self.0
    }

    #[inline]
    pub unsafe fn to_block(&self) -> NonNull<Block> {
        let raw = self as *const _ as *const u8;
        let header_offset = (self.0 as isize) * -1;
        NonNull::new_unchecked(raw.offset(header_offset) as *mut Block)
    }
}


/// This struct maintains an ordered tree of free blocks.
///
/// The structure internally contains two red-black trees, one sorted by address order,
/// and the other by a user-selected sort order, this sorting determines the
/// order in which free blocks are selected for use by a carrier during allocation.
pub struct FreeBlockTree {
    // address-ordered tree
    atree: RBTree<SortedKeyAdapter<FreeBlock<RBTreeLink>>>,
    // custom-ordered tree
    stree: RBTree<SortedKeyAdapter<FreeBlock<RBTreeLink>>>,
    // the ordering used by `stree`
    order: SortOrder,
}
impl FreeBlockTree {
    #[inline]
    pub fn new(order: SortOrder) -> Self {
        Self {
            atree: RBTree::new(SortedKeyAdapter::new(SortOrder::AddressOrder)),
            stree: RBTree::new(SortedKeyAdapter::new(order)),
            order,
        }
    }

    /// Lookup a free block which is the best fit for the given requested size
    pub fn find_best_fit(&mut self, layout: &Layout) -> Option<&FreeBlock<RBTreeLink>> {
        let mut cursor = self.stree.front();
        let mut result = None;
        let mut best_size = 0;

        let aligned = layout.pad_to_align().unwrap();
        let requested = aligned.size();

        match self.order {
            SortOrder::AddressOrder => {
                while let Some(block) = cursor.get() {
                    // When in AddressOrder, we have to search the whole tree for the best fit
                    let usable = unsafe { block.usable_size() };

                    // Not suitable
                    if usable < requested {
                        cursor.move_next();
                        continue;
                    }

                    // If we've found a better fit, or don't yet have a fit
                    // mark the current node as the current best fit
                    if usable < best_size || result.is_none() {
                        result = Some(block);
                        best_size = usable;
                    }

                    cursor.move_next();
                }
            }
            SortOrder::SizeAddressOrder => {
                while let Some(block) = cursor.get() {
                    // A best fit can be found as the previous neighbor of the first block which is too small
                    // or the last block in the tree, if all blocks are of adequate size
                    let usable = unsafe { block.usable_size() };
                    if usable < requested {
                        break;
                    }
                    result = Some(block);
                    cursor.move_next();
                }
            }
        }

        result
    }

    /// Inserts the given block into this tree
    pub unsafe fn insert(&mut self, block: *const FreeBlock<RBTreeLink>) {
        let _ = self.atree.insert(UnsafeRef::from_raw(block));
        let _ = self.stree.insert(UnsafeRef::from_raw(block));
    }

    /// Removes the given block from this tree
    pub unsafe fn remove(&mut self, block: *const FreeBlock<RBTreeLink>) {
        // remove from address-ordered tree
        let mut cursor = self.atree.cursor_mut_from_ptr(block);
        let removed = cursor.remove();
        assert!(removed.is_some());

        // remove from user-ordered tree
        let mut cursor = self.stree.cursor_mut_from_ptr(block);
        let removed = cursor.remove();
        assert!(removed.is_some());
    }
}

#[cfg(test)]
mod tests {
    use crate::mmap;
    use crate::sys;
    use super::*;

    #[test]
    fn basic_block_api() {
        let size = sys::pagesize();
        let usable = size - mem::size_of::<Block>();
        let layout = Layout::from_size_align(size, size).unwrap();
        let ptr = unsafe { mmap::map(layout.clone()).expect("unable to map memory") };
        let raw = ptr.as_ptr() as *mut Block;
        unsafe {
            ptr::write(
                raw,
                Block(usable),
            );
        }
        let mut block = unsafe {  &mut *raw };
        // Set attributes for this block
        block.set_allocated();
        assert!(!block.is_free());
        block.set_free();
        assert!(block.is_free());
        block.set_last();
        assert!(block.is_last());
        block.clear_last();
        assert!(!block.is_last());
        block.set_last();
        assert!(!block.is_prev_free());
        block.set_prev_free();
        assert!(block.is_prev_free());
        block.set_prev_allocated();
        // Make sure size matches
        assert_eq!(block.usable_size(), usable);
        // Make sure we don't read uninitialized memory
        assert_eq!(block.next(), None);

        // Cleanup
        unsafe { mmap::unmap(raw as *mut u8, layout) };
    }

    #[test]
    fn block_try_alloc() {
        let size = sys::pagesize();
        let usable = size - mem::size_of::<Block>();
        let layout = Layout::from_size_align(size, size).unwrap();
        let ptr = unsafe { mmap::map(layout.clone()).expect("unable to map memory") };
        let raw = ptr.as_ptr() as *mut Block;
        unsafe {
            ptr::write(
                raw,
                Block(usable),
            );
        }
        let mut block = unsafe { &mut *raw };
        // Block is free, and last
        block.set_free();
        block.set_last();
        assert_eq!(block.usable_size(), usable);
        // Try allocate entire usable size
        let request_layout = Layout::from_size_align(usable, mem::size_of::<usize>()).unwrap();
        let result = block.try_alloc(&request_layout);
        assert!(result.is_ok());
        // Block should no longer be free
        assert!(!block.is_free());
        // Another attempt to allocate this block should fail
        let result = block.try_alloc(&request_layout);
        assert!(result.is_err());
        // Free the block
        block.free();
        assert!(block.is_free());
        // Should have a block footer now
        let result = block.footer();
        assert!(result.is_some());
        let result = result.unwrap();
        let footer = unsafe { result.as_ref() };
        assert_eq!(footer.usable_size(), usable);
        // Another allocation in this block will succeed
        let result = block.try_alloc(&request_layout);
        assert!(result.is_ok());

        // Cleanup
        unsafe { mmap::unmap(raw as *mut u8, layout) };
    }

    #[test]
    fn free_block_tree_test() {
        // Allocate space for two blocks, each page sized, but we're going to treat
        // the latter block as half that size
        let size = sys::pagesize() * 2;
        let usable = sys::pagesize() - mem::size_of::<Block>();
        let layout = Layout::from_size_align(size, size).unwrap();
        let ptr = unsafe { mmap::map(layout.clone()).expect("unable to map memory") };
        // Get pointers to both blocks
        let raw = ptr.as_ptr() as *mut FreeBlock<RBTreeLink>;
        let raw2 = unsafe { (raw as *mut u8).offset(sys::pagesize() as isize) as *mut FreeBlock<RBTreeLink> };
        // Write block headers
        unsafe {
            let mut block1 = Block(usable);
            block1.set_free();
            let mut block2 = Block(usable / 2);
            block2.set_free();
            block2.set_prev_free();
            block2.set_last();
            ptr::write(
                raw,
                FreeBlock {
                    header: block1,
                    addr_link: RBTreeLink::new(),
                    user_link: RBTreeLink::new(),
                }
            );
            ptr::write(
                raw2,
                FreeBlock {
                    header: block2,
                    addr_link: RBTreeLink::new(),
                    user_link: RBTreeLink::new(),
                }
            );
        }
        // Get blocks
        let fblock1 = raw;
        let fblock2 = raw2;
        // Need a sub-region here so we can clean up the mapped memory at the end
        // without a segmentation fault due to dropping the tree after the memory
        // is unmapped
        {
            // Create empty tree
            let mut tree = FreeBlockTree::new(SortOrder::SizeAddressOrder);
            // Add blocks to tree
            unsafe {
                tree.insert(fblock1);
                tree.insert(fblock2);
            }
            // Find free block, we should get fblock2, since best fit would be the
            // smallest block which fits the request, and our tree should be sorted by
            // size
            let req_size = 1024;
            let request_layout = Layout::from_size_align(req_size, mem::size_of::<usize>()).unwrap();
            let result = tree.find_best_fit(&request_layout);
            assert!(result.is_some());
            let result_block = result.unwrap();
            assert_eq!(result_block as *const _ as *const u8, fblock2 as *const u8);
        }
        unsafe { mmap::unmap(raw as *mut u8, layout) };
    }
}
