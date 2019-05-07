// TODO: WIP
// The following is ported from Go, and it's generated size class table
//
// https://github.com/golang/go/blob/a62887aade0fa0db4c4bd47aed72d93cb820da2c/src/runtime/mksizeclasses.go
use core::ptr;

use crate::sys;

// The largest size class
const MAX_SIZE_CLASS: usize = 32768;
// The object size for large size classes
const LARGE_SIZE_DIV: usize = 128;
// The object size for small size classes
const SMALL_SIZE_DIV: usize = 8;
// The maximum size that is considered in the small size class
const SMALL_SIZE_MAX: usize = 1024;
// The number of size classes defined
const NUM_CLASSES: usize = 67;

// The byte pattern used for small size class blocks
#[cfg(debug_assertions)]
const SMALL_CLASS_PATTERN: u8 = 0x35;
// The byte pattern used for large size class blocks
#[cfg(debug_assertions)]
const LARGE_CLASS_PATTERN: u8 = 0x57;

// The table of size classes, or really the sizes of the classes
const CLASS_TO_SIZE: [usize; NUM_CLASSES] = [
    0, 8, 16, 32, 48, 64, 80, 96, 112, 128,
    144, 160, 176, 192, 208, 224, 240, 256,
    288, 320, 352, 384, 416, 448, 480, 512,
    576, 640, 704, 768, 896, 1024, 1152, 1280,
    1408, 1536, 1792, 2048, 2304, 2688, 3072, 3200,
    3456, 4096, 4864, 5376, 6144, 6528, 6784, 6912,
    8192, 9472, 9728, 10240, 10880, 12288, 13568, 14336,
    16384, 18432, 19072, 20480, 21760, 24576, 27264, 28672,
    32768
];

// This table maps the size classes to the number of pages to allocate
// when allocating a new block of memory for objects of a given size class,
// i.e. when allocating a new block for a size class of 8 bytes, allocate 1
// page from the system, and then divide that page up into 8 byte chunks
const CLASS_TO_NUM_PAGES: [usize; NUM_CLASSES] = [
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    2, 1, 2, 1, 2, 1, 3, 2,
    3, 1, 3, 2, 3, 4, 5, 6,
    1, 7, 6, 5, 4, 3, 5, 7,
    2, 9, 7, 5, 8, 3, 10, 7,
    4
];

// This table maps a small class request to the size class for that request
const SIZE_TO_SMALL_CLASS: [usize; SMALL_SIZE_MAX / SMALL_SIZE_DIV + 1] = [
    0, 1, 2, 3, 3, 4, 4, 5, 5, 6,
    6, 7, 7, 8, 8, 9, 9, 10, 10, 11,
    11, 12, 12, 13, 13, 14, 14, 15, 15, 16,
    16, 17, 17, 18, 18, 18, 18, 19, 19, 19,
    19, 20, 20, 20, 20, 21, 21, 21, 21, 22,
    22, 22, 22, 23, 23, 23, 23, 24, 24, 24,
    24, 25, 25, 25, 25, 26, 26, 26, 26, 26,
    26, 26, 26, 27, 27, 27, 27, 27, 27, 27,
    27, 28, 28, 28, 28, 28, 28, 28, 28, 29,
    29, 29, 29, 29, 29, 29, 29, 30, 30, 30,
    30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    30, 30, 30, 31, 31, 31, 31, 31, 31, 31,
    31, 31, 31, 31, 31, 31, 31, 31, 31
];

// This table maps a large class request to the size class for that request
const SIZE_TO_LARGE_CLASS: [usize; (MAX_SIZE_CLASS - SMALL_SIZE_MAX) / LARGE_SIZE_DIV + 1] = [
    31, 32, 33, 34, 35, 36, 36, 37, 37, 38,
    38, 39, 39, 39, 40, 40, 40, 41, 42, 42,
    43, 43, 43, 43, 43, 44, 44, 44, 44, 44,
    44, 45, 45, 45, 45, 46, 46, 46, 46, 46,
    46, 47, 47, 47, 48, 48, 49, 50, 50, 50,
    50, 50, 50, 50, 50, 50, 50, 51, 51, 51,
    51, 51, 51, 51, 51, 51, 51, 52, 52, 53,
    53, 53, 53, 54, 54, 54, 54, 54, 55, 55,
    55, 55, 55, 55, 55, 55, 55, 55, 55, 56,
    56, 56, 56, 56, 56, 56, 56, 56, 56, 57,
    57, 57, 57, 57, 57, 58, 58, 58, 58, 58,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    58, 59, 59, 59, 59, 59, 59, 59, 59, 59,
    59, 59, 59, 59, 59, 59, 59, 60, 60, 60,
    60, 60, 61, 61, 61, 61, 61, 61, 61, 61,
    61, 61, 61, 62, 62, 62, 62, 62, 62, 62,
    62, 62, 62, 63, 63, 63, 63, 63, 63, 63,
    63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
    63, 63, 63, 63, 63, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 65, 65, 65, 65,
    65, 65, 65, 65, 65, 65, 65, 66, 66, 66,
    66, 66, 66, 66, 66, 66, 66, 66, 66, 66,
    66, 66, 66, 66, 66, 66, 66, 66, 66, 66,
    66, 66, 66, 66, 66, 66, 66, 66, 66
];

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
#[repr(packed)]
pub struct SlabCarrier<L: Link> {
    header: usize,
    link: L,
    blocks: BitVec,
}
impl<L: Link> SlabCarrier<L> {
    // The number of bits to shift/mask to find a superaligned address
    const SA_BITS: usize = 18;
    // The number of bits to shift to find a superaligned carrier address
    const SA_CARRIER_SHIFT: usize = SA_BITS;
    // The size of a superaligned carrier, 262k (262,144 bytes)
    const SA_CARRIER_SIZE: usize = 1usize << SA_CARRIER_SHIFT;
    // The mask needed to go from a pointer in a SA carrier to the carrier
    const SA_CARRIER_MASK: usize = (!0usize) << SA_CARRIER_SHIFT;
    // The pattern for bytes written to blocks that are freed
    const FREE_PATTERN: u8 = 0x57;

    #[inline]
    pub fn new(size_class: usize) -> Self {
        let num_blocks = SA_CARRIER_SIZE / size_class;
        Self {
            header: size_class,
            link: L::default(),
            blocks: BitVec::with_capacity(num_blocks * 8),
        }
    }

    pub unsafe fn alloc_block(&mut self) -> Result<NonNull<u8>, AllocErr> {
        let mut index = 0;
        for allocated in self.blocks.iter() {
            if allocated {
                index++;
                continue;
            }

            // Get pointer to start of carrier
            let first_block = self.head();
            // Calculate selected block address
            let block_size = self.header;
            // NOTE: If `index` is 0, the first block was selected
            let block = first_block.offset(block_size * index);
            // Mark allocated
            self.blocks.set(index, true);
            // Return pointer to block
            return Ok(NonNull::new_unchecked(block));
        }

        // No space available
        Err(AllocErr)
    }

    pub unsafe fn free_block(&mut self, ptr: *const u8) {
        let index = ((ptr as usize) - (first_block as usize)) / 8;
        assert!(index >= bv_size);

        if cfg!(debug_assertions) {
            // Get first block
            let first_block = self.head();
            let block_size = self.header;
            let block = first_block.offset(block_size * index);

            // Write free pattern over block
            ptr::write_bytes(block, Self::FREE_PATTERN, block_size);
        }

        // Mark block as free
        self.blocks.set(index, false);
    }

    #[inline]
    fn head(&self) -> *const u8 {
        // Get carrier pointer
        let carrier = self as *const _ as *const u8;
        // Calculate header size
        let meta_size = mem::size_of::<usize>() + mem::size_of::<L>();
        let bv_size = self.blocks.capacity() / 8;
        let header_size = meta_size + bv_size;
        // NOTE: All size classes are word-aligned
        let layout = Layout::from_size_align_unchecked(header_size, mem::size_of::<usize>())
            .pad_to_align()
            .unwrap();
        let header_size = layout.size();
        // Calculate address of first block
        carrier.offset(header_size)
    }
}

// Type alias for the list of currently allocated single-block carriers
type SlabCarrierList = LinkedList<SlabCarrierListAdapter>;

// Implementation of adapter for intrusive collection used for slab carriers
intrusive_adapter!(SlabCarrierListAdapter = UnsafeRef<SlabCarrier>: SlabCarrier { link: LinkedListLink });

// This struct maintains the actual size class buckets, we use the above
// defined constants to map requests to either a bucket which should serve
// the request, or indicate to the caller that the request is too large and
// should be allocated directly from the system
pub struct SizeClasses([*const SlabCarrierList; NUM_CLASSES]);

impl SizeClasses {
    // Creates a new empty instance of the struct
    #[inline]
    fn new() -> Result<SizeClasses, AllocErr> {
        let buckets = [ptr::null_mut(); NUM_CLASSES];
        let mut index = 1;
        for bucket in buckets {
            let size_class = CLASS_TO_SIZE[index];
            buckets[index] = Self::create_carrier(size_class)?;
        }

        Ok(SizeClasses(buckets))
    }

    // Gets the size class bucket which can serve an allocation of the requested
    // size, or returns `None` to indicate that the request should allocate directly
    // from the system
    pub fn get(&self, size: usize) -> Option<*const SlabCarrier> {
        // If too large, there is no size class, such allocations are performed directly from the system
        if size > MAX_SIZE_CLASS {
            return None;
        }

        // Size classes are divided into small and large categories, this bit
        // helps us find the correct size class depending on the category the requested
        // size falls in
        let class = if size <= SMALL_SIZE_MAX - 8 {
            SIZE_TO_SMALL_CLASS[(size + SMALL_SIZE_DIV - 1) / SMALL_SIZE_DIV]
        } else {
            SIZE_TO_LARGE_CLASS[(size - SMALL_SIZE_MAX + LARGE_SIZE_DIV - 1) / LARGE_SIZE_DIV]
        };

        let bucket = self.0[class];

        // If the bucket has no free blocks, allocate some is possible
        if bucket.is_null() {
            return self.get_and_fill_bucket(class);
        }

        Some(bucket)
    }

    #[inline]
    fn get_and_fill_bucket(&self, class: usize) -> Option<*const FreeBlock<'a>> {
        use core::intrinsics;

        // The size of each block in this class
        let class_size = CLASS_TO_SIZE[class];
        // The number of pages to alloc from the system for this class
        let num_pages = CLASS_TO_NUM_PAGES[class];
        // The total allocation size in bytes
        let psize = num_pages * sys::pagesize();

        // Allocate pages from the system, word-aligned
        let layout = Layout::from_size_align_unchecked(psize, mem::size_of::<usize>());
        match sys::alloc(layout) {
            Err(_) => return None,
            Ok(ptr) => {
                let raw = ptr.as_ptr();
                // Now we need to create N blocks from P pages, where N is P/class_size
                let num_blocks = num_pages / class_size;
                // Track the previously created block so we can set the neighbor links
                let mut prev: *const FreeBlock = ptr::null_mut();

                // For each block of N blocks..
                for i in 0..num_blocks {
                    // The current block pointer is given by offsetting `raw` by `i * class_size` bytes
                    let current = intrinsics::arith_offset(raw, i * class_size) as *mut FreeBlock;
                    // Write the block header to the beginning of the block
                    ptr::write(
                        current,
                        FreeBlock {
                            header: BlockHeader::default(),
                        }
                    );

                    // Initialize the block data region with a special pattern which can help
                    // identify issues with allocators or misuse of raw pointers. This is only
                    // done when debug assertions are enabled
                    write_free_pattern(current, class_size);

                    // If we have a previously created block, set the current block as its next
                    // neighbor, then update `prev` to point to `current` for the next block
                    if !prev.is_null() {
                        prev.header.neighbors.set_next(block_ptr);
                    }

                    prev = current;
                }

                // "Fill" the bucket
                let head = raw as *const FreeBlock;
                self.0[class] = head;

                Some(head)
            }
        }
    }
}

#[cfg(debug_assertions)]
fn write_free_pattern(block: *mut FreeBlock, size: usize) {
    use core::mem::size_of;

    // The data region of a block is given by offsetting the block pointer
    // by the size of the FreeBlock structure.
    let data = unsafe { block.offset(1) as *mut u8 };
    // The resulting pointer should be word aligned
    assert_word_aligned!(data);

    // Match the block size class to the appropriate byte pattern
    let pattern = if size > SMALL_SIZE_MAX {
        SMALL_CLASS_PATTERN
    } else {
        LARGE_CLASS_PATTERN
    };

    // Write the pattern over all bytes in the block except those
    // containing the block header itself
    ptr::write_bytes(data, pattern, (size - size_of::<FreeBlock>()));
}

#[cfg(not(debug_assertions))]
fn write_free_pattern(_block: *mut FreeBlock, _class: usize) {}
