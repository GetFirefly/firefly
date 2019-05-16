use core::alloc::AllocErr;
use core::mem;
use core::ptr::{self, NonNull};

use liblumen_core::alloc::size_classes::SizeClass;

use crate::blocks::BlockBitSet;
use crate::sorted::Link;

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
pub struct SlabCarrier<L, B> {
    header: usize,
    pub(crate) link: L,
    blocks: B,
}
impl<L, B> SlabCarrier<L, B>
where
    L: Link,
    B: BlockBitSet,
{
    // The pattern for bytes written to blocks that are freed
    const FREE_PATTERN: u8 = 0x57;

    /// Initializes a `SlabCarrier` using the memory indicated by
    /// the given pointer and size as the slab it will manage. The
    /// size class is assigned to all blocks in this carrier
    #[inline]
    pub unsafe fn init(ptr: *mut u8, size: usize, size_class: SizeClass) -> *mut Self {
        let size_class_bytes = size_class.to_bytes();
        // Shift pointer past end of Self
        let data_ptr = (ptr as *mut Self).offset(1) as *mut u8;
        // Pointer to the block set is given by offset backwards to the beginning of the field
        let abs_ptr = data_ptr.offset(-1 * mem::size_of::<B>() as isize) as *mut B;
        // We write the carrier header to memory, as well as the block set
        // NOTE: The block set actually gets written first, as part of `new/3`
        ptr::write(
            ptr as *mut Self,
            Self {
                header: size_class_bytes,
                link: L::default(),
                blocks: B::new(abs_ptr, size, size_class_bytes),
            },
        );
        ptr as *mut Self
    }

    /// Returns the number of free blocks in this carrier
    #[allow(unused)]
    #[inline]
    pub fn available_blocks(&self) -> usize {
        self.blocks.count_free()
    }

    /// Allocates a block within this carrier, if one is available
    pub unsafe fn alloc_block(&self) -> Result<NonNull<u8>, AllocErr> {
        for (index, allocated) in self.blocks.iter().enumerate() {
            if allocated {
                continue;
            }

            if self.blocks.try_alloc(index) {
                // We were able to mark this block allocated
                // Get pointer to start of carrier
                let first_block = self.head();
                // Calculate selected block address
                let block_size = self.header;
                // NOTE: If `index` is 0, the first block was selected
                let block = first_block.offset((block_size * index) as isize);
                // Return pointer to block
                return Ok(NonNull::new_unchecked(block));
            }
        }

        // No space available
        Err(AllocErr)
    }

    /// Deallocates a block within this carrier
    pub unsafe fn free_block(&self, ptr: *mut u8) {
        let first_block = self.head() as *mut u8;
        let index = ((ptr as usize) - (first_block as usize)) / 8;
        debug_assert!(index < self.blocks.size());

        if cfg!(debug_assertions) {
            // Write free pattern over block
            let block_size = self.header;
            ptr::write_bytes(ptr, Self::FREE_PATTERN, block_size);
        }

        // Mark block as free
        self.blocks.free(index);
    }

    #[inline]
    fn head(&self) -> *mut u8 {
        // Get carrier pointer
        let carrier = self as *const _ as *mut u8;
        // Shift pointer past header
        let offset = mem::size_of::<Self>() + self.blocks.extent_size();
        unsafe { carrier.offset(offset as isize) }
    }
}
