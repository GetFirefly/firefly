use core::mem;
use core::ptr::{self, NonNull};

use liblumen_core::alloc::AllocErr;
use liblumen_core::alloc::size_classes::SizeClass;

use crate::blocks::{BlockBitSet, BlockBitSubset};
use crate::sorted::Link;
use std::marker::PhantomData;

// The slab carrier, like the more general `MultiBlockCarrier`,
// is allocated in a super-aligned region, giving 262k of space
// for allocating blocks. Due to the carrier header, this size
// is actually reduced by two words, 128 bits on 64bit architectures,
// 64 bits on 32bit architectures. The header contains the size
// class index and the offset
//
// With the remaining space in that region,
// - 1 word
// of space to divide up into blocks within the region. For the
// largest size class of 32k, that means we can fit 7 blocks,
// while the smallest size class of 8 bytes can fit 32,767 blocks.
//
//
// Carrier header contains two pieces of information
// - 8 bits for size class index (maximum value is 67)
// - 16 bits for offset
pub struct SlabCarrier<L, S> {
    block_byte_len: usize,
    pub(crate) link: L,
    block_bit_subset_type: PhantomData<S>,
}
impl<L, S> SlabCarrier<L, S>
where
    L: Link,
    S: BlockBitSubset,
{
    // The pattern for bytes written to blocks that are freed
    const FREE_PATTERN: u8 = 0x57;

    pub fn can_fit_multiple_blocks(byte_len: usize, size_class: &SizeClass) -> bool {
        BlockBitSet::<S>::can_fit_multiple_blocks(byte_len, size_class.to_bytes())
    }

    /// Initializes a `SlabCarrier` using the memory indicated by
    /// the given `ptr` and `byte_len` as the slab it will manage. The
    /// `size_class` is assigned to all blocks in this carrier
    #[inline]
    pub unsafe fn init(ptr: *mut u8, byte_len: usize, size_class: SizeClass) -> *mut Self {
        let size_class_byte_len = size_class.to_bytes();
        assert!(size_class_byte_len < byte_len);
        let self_ptr = ptr as *mut Self;
        self_ptr.write(Self {
            block_byte_len: size_class_byte_len,
            link: L::default(),
            block_bit_subset_type: PhantomData,
        });
        // Shift pointer past end of Self
        let block_bit_set_ptr = self_ptr.add(1) as *mut BlockBitSet<S>;
        BlockBitSet::write(block_bit_set_ptr, byte_len, size_class_byte_len);

        self_ptr
    }

    fn block_bit_set(&self) -> &BlockBitSet<S> {
        let self_ptr = self as *const Self;

        unsafe {
            let block_bit_set_ptr = self_ptr.add(1) as *mut BlockBitSet<S>;

            &*block_bit_set_ptr
        }
    }

    /// Returns the number of free blocks in this carrier
    #[allow(unused)]
    #[inline]
    pub fn available_blocks(&self) -> usize {
        self.block_bit_set().count_free()
    }

    /// Allocates a block within this carrier, if one is available
    pub unsafe fn alloc_block(&self) -> Result<NonNull<u8>, AllocErr> {
        match self.block_bit_set().alloc_block() {
            Ok(index) => {
                // We were able to mark this block allocated
                // Get pointer to start of carrier
                let first_block = self.head();
                // Calculate selected block address
                let block_size = self.block_byte_len;
                // NOTE: If `index` is 0, the first block was selected
                let block = first_block.add(block_size * index);

                // Return pointer to block
                Ok(NonNull::new_unchecked(block))
            }
            // No space available
            Err(AllocErr) => Err(AllocErr),
        }
    }

    /// Deallocates a block within this carrier
    pub unsafe fn free_block(&self, ptr: *mut u8) {
        // Get pointer to start of carrier
        let first_block = self.head() as *mut u8;
        // Get block size in order to properly calculate index of block
        let block_size = self.block_byte_len;
        // By subtracting the pointer we got from the base pointer, and
        // dividing by the block size, we get the index of the block
        let offset = ptr.offset_from(first_block);
        assert!(
            0 <= offset,
            "ptr ({:p}) is before first_block ({:p})",
            ptr,
            first_block
        );
        let add: usize = offset as usize;

        let misalignment = add % block_size;
        assert_eq!(
            misalignment,
            0,
            "ptr ({:p}) is not a multiple of block_size ({:?}) after first_block {:p}.  Misaligned by {:?}.",
            ptr,
            block_size,
            first_block,
            misalignment
        );

        let index = add / block_size;
        // The index should always be less than the size
        debug_assert!(
            index < self.block_bit_set().len(),
            "index ({:?}) exceeds block bit set length ({:?})",
            index,
            self.block_bit_set().len()
        );

        if cfg!(debug_assertions) {
            // Write free pattern over block
            ptr::write_bytes(ptr, Self::FREE_PATTERN, block_size);
        }

        // Mark block as free
        self.block_bit_set().free(index);
    }

    #[inline]
    fn head(&self) -> *mut u8 {
        // Get carrier pointer
        let carrier = self as *const _ as *mut u8;
        // Shift pointer past header
        let offset = mem::size_of::<Self>() + self.block_bit_set().size();
        unsafe { carrier.add(offset) }
    }
}
