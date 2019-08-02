use core::mem;
use core::ptr::{self, NonNull};

use alloc::fmt::{self, Debug, Formatter};
use alloc::string::String;

use liblumen_core::alloc::alloc_utils;

use super::{BlockFooter, BlockRef, FreeBlock, FreeBlockRef};

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
#[derive(Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct Block(usize);

// Helpers for Debug implementation
impl Block {
    #[cfg(target_pointer_width = "32")]
    #[inline(always)]
    pub(crate) fn format_raw(&self) -> String {
        format!("{:032b}", self.0)
    }

    #[cfg(target_pointer_width = "64")]
    #[inline(always)]
    pub(crate) fn format_raw(&self) -> String {
        format!("{:064b}", self.0)
    }
}

impl Debug for Block {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("Block")
            .field("size", &self.usable_size())
            .field("free", &self.is_free())
            .field("last", &self.is_last())
            .field("prev_free", &self.is_prev_free())
            .field("raw", &self.format_raw())
            .field("address", &alloc_utils::format_address_of(self))
            .finish()
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

    /// Creates a new Block with the given size
    #[inline(always)]
    pub fn new(size: usize) -> Self {
        assert!(size >= FreeBlock::min_block_size());
        Self(size)
    }

    /// Gets a raw pointer to the data region of this block
    #[inline]
    pub unsafe fn data(&self) -> *const u8 {
        let raw = self as *const Block;
        raw.add(1) as *const u8
    }

    /// Checks if the given pointer belongs to this block
    #[inline]
    pub fn owns(&self, ptr: *const u8) -> bool {
        let usable = self.usable_size();
        // Get pointer to start of block
        let this = self as *const _ as *const u8 as usize;
        // Get pointer to end of block
        let end = this + mem::size_of::<Block>() + usable;
        // Check if pointer is in block's owned range
        let ptr = ptr as usize;
        ptr >= this && ptr <= end
    }

    /// Gets the usable size of this block in bytes
    #[inline]
    pub fn usable_size(&self) -> usize {
        self.0 & !Self::HIGH_BIT_MASK
    }

    /// Updates the size metadata of this block in bytes
    ///
    /// NOTE: This doesn't actually change the amount of usable memory
    #[inline(always)]
    pub fn set_size(&mut self, new_size: usize) {
        assert!(new_size >= FreeBlock::min_block_size());
        self.0 = (self.0 & Self::HIGH_BIT_MASK) | new_size;
    }

    /// Determines if this block is free or not
    #[inline(always)]
    pub fn is_free(&self) -> bool {
        (self.0 & Self::FREE_FLAG) == Self::FREE_FLAG
    }

    /// Marks this block as free by setting the appropriate flag bit
    #[inline(always)]
    pub fn set_free(&mut self) {
        self.0 |= Self::FREE_FLAG;
    }

    /// Marks this block as allocated by clearing the free flag bit
    #[inline(always)]
    pub fn set_allocated(&mut self) {
        self.0 &= !Self::FREE_FLAG;
    }

    /// Determines if the previous neighboring block is free or not
    #[inline(always)]
    pub fn is_prev_free(&self) -> bool {
        (self.0 & Self::PREV_FREE_FLAG) == Self::PREV_FREE_FLAG
    }

    /// Marks the flag bit which indicates that the previous neighboring block is free
    #[inline(always)]
    pub fn set_prev_free(&mut self) {
        self.0 |= Self::PREV_FREE_FLAG
    }

    /// Clears the flag bit which indicates that the previous neighboring block is free
    #[allow(unused)]
    #[inline(always)]
    pub fn set_prev_allocated(&mut self) {
        self.0 &= !Self::PREV_FREE_FLAG
    }

    /// Determines if this block is the last block in its containing memory region
    #[inline(always)]
    pub fn is_last(&self) -> bool {
        (self.0 & Self::LAST_FLAG) == Self::LAST_FLAG
    }

    /// Marks this block as the last block by setting the appropriate flag bit
    #[inline(always)]
    pub fn set_last(&mut self) {
        self.0 |= Self::LAST_FLAG;
    }

    /// Clears the flag bit which marks this block as the last block
    #[inline(always)]
    pub fn clear_last(&mut self) {
        self.0 &= !Self::LAST_FLAG;
    }

    /// Locates the next block following this block, if it exists.
    #[inline]
    pub fn next(&self) -> Option<BlockRef> {
        if self.is_last() {
            return None;
        }

        let ptr = self as *const Block;
        let data_ptr = unsafe { ptr.add(1) as *mut u8 };
        let next_ptr = unsafe { data_ptr.add(self.usable_size()) };

        Some(unsafe { BlockRef::from_raw(next_ptr as *mut Block) })
    }

    /// Locates the previous block, if it exists and is free.
    ///
    /// If the previous free flag is set, we know we can shift back one word to
    /// access the BlockFooter for that free block. This can be accessed to give
    /// us the usable size of the previous block, which combined with the static
    /// size of the Block type, gives us the offset needed to calculate the pointer
    /// to that block.
    #[inline]
    pub fn prev(&self) -> Option<FreeBlockRef> {
        if !self.is_prev_free() {
            return None;
        }

        let this = self as *const _ as *const BlockFooter;
        let prev_footer_ptr = unsafe { this.offset(-1) as *const usize };
        let usable = unsafe { *prev_footer_ptr };
        let offset = -1isize * (usable + mem::size_of::<Block>()) as isize;
        let prev_ptr = unsafe { (prev_footer_ptr as *const u8).offset(offset) as *mut Block };

        Some(unsafe { FreeBlockRef::from_raw(prev_ptr as *mut FreeBlock) })
    }

    /// Locates the block footer for this block, if it exists.
    ///
    /// NOTE: This function returns an Option because a footer is only
    /// present when the block is free,
    #[allow(unused)]
    #[inline]
    pub fn footer(&self) -> Option<NonNull<BlockFooter>> {
        if !self.is_free() {
            return None;
        }

        let size = self.usable_size();
        let offset = size - mem::size_of::<BlockFooter>();
        unsafe {
            let ptr = self.data();
            let footer_ptr = ptr.add(offset) as *mut BlockFooter;
            Some(NonNull::new_unchecked(footer_ptr))
        }
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
    /// the freed block, but it *does not* initialize the links in `FreeBlock`,
    /// since we don't know the type of the link here. Instead it is up to
    /// the carrier to initialize the links as part of the work it performs during
    /// deallocation of a block.
    #[inline]
    pub fn free(&mut self) -> FreeBlockRef {
        // Freeing a block here only means marking this block as free,
        // the higher level carrier code is responsible for coalescing
        // blocks and releasing memory as appropriate

        // Set free flag
        self.set_free();
        // If not last, update the neighboring block to indicate that
        // this block is free, so that coalescing operations can join them
        if !self.is_last() {
            let mut next_ref = self.next().expect("expected neighboring block");
            let next = next_ref.as_mut();
            next.set_prev_free();
        }
        // Write free pattern over this block, a no-op when not in debug mode
        self.write_free_pattern();
        // Add block footer
        let size = self.usable_size();
        let offset = size - mem::size_of::<BlockFooter>();
        unsafe {
            let ptr = self.data();
            let footer_ptr = ptr.add(offset) as *mut BlockFooter;
            ptr::write(footer_ptr, BlockFooter::new(size));
        }
        // Ensure links are initialized
        let this = self as *mut Block;
        let this_free = this as *mut FreeBlock;
        unsafe {
            ptr::write(this_free, FreeBlock::from_block(self.clone()));
            FreeBlockRef::from_raw(this_free)
        }
    }

    /// Use with care, should only be written over free blocks
    #[cfg(debug_assertions)]
    pub(crate) fn write_free_pattern(&self) {
        // The data pointer must be word aligned
        let mut ptr = self as *const _ as *mut u8;
        let mut len = self.usable_size();
        if self.is_free() {
            ptr = unsafe { ptr.add(mem::size_of::<FreeBlock>()) };
            // Usable size is total size - sizeof(Block), but a free block
            // is total size - sizeof(FreeBlock) - sizeof(BlockFooter)
            len = (len + mem::size_of::<Block>())
                - mem::size_of::<FreeBlock>()
                - mem::size_of::<BlockFooter>();
        } else {
            ptr = unsafe { ptr.add(mem::size_of::<Block>()) };
        }
        assert_word_aligned!(ptr);

        // Write the pattern over all bytes in the block except those
        // containing the block header itself
        unsafe {
            ptr::write_bytes(ptr, Self::FREE_PATTERN, len);
        }
    }

    /// Use with care, should only be written over free blocks
    #[cfg(not(debug_assertions))]
    pub(crate) fn write_free_pattern(&self) {}
}
