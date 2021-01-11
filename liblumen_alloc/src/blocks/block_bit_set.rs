use core::marker::PhantomData;
use core::mem;
use core::sync::atomic::{AtomicUsize, Ordering};

use liblumen_core::alloc::AllocError;

use crate::mem::bit_size_of;

pub trait BlockBitSubset: Default {
    /// Try to allocate block in first `bit_len` bits of subset.
    ///
    /// NOTE: This operation can fail, primarily in multi-threaded scenarios
    /// with atomics in play. If false is returned, it means that either the block was
    /// already allocated in the span of time between when you looked up the block
    /// and when you tried to allocate it, or a neighboring bit in the subset was flipped. You may
    /// retry, or try searching for another block, or simply fail the allocation request. It is up
    /// to the allocator.
    fn alloc_block(&self, bit_len: usize) -> Result<usize, AllocError>;

    /// Return a count of the allocated blocks managed by this subset
    fn count_allocated(&self) -> usize;

    /// Free the block represented by the given bit index
    fn free(&self, bit: usize);
}

#[derive(Default)]
#[repr(transparent)]
pub struct ThreadSafeBlockBitSubset(AtomicUsize);

impl BlockBitSubset for ThreadSafeBlockBitSubset {
    fn alloc_block(&self, bit_len: usize) -> Result<usize, AllocError> {
        assert!(bit_len <= bit_size_of::<Self>());

        // On x86 this could use `bsf*` (Bit Scan Forward) instructions
        for i in 0..bit_len {
            let flag = 1usize << i;
            let current = self.0.load(Ordering::Acquire);

            if current & flag == flag {
                // Already allocated
                continue;
            }

            if self
                .0
                .compare_exchange(current, current | flag, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(i);
            }
        }

        Err(AllocError)
    }

    fn count_allocated(&self) -> usize {
        self.0.load(Ordering::Acquire).count_ones() as usize
    }

    fn free(&self, bit: usize) {
        let flag = 1usize << bit;
        self.0.fetch_and(!flag, Ordering::AcqRel);
    }
}

unsafe impl Sync for ThreadSafeBlockBitSubset {}
unsafe impl Send for ThreadSafeBlockBitSubset {}

/// This trait abstracts out the concept of a bit set which tracks
/// free/allocated blocks within some contiguous region of memory.
///
/// The actual implementation may differ based on whether it uses atomics
/// or not, and what granularity is used for the bit representation, i.e.
/// u8, u16, usize, etc.
pub struct BlockBitSet<S: BlockBitSubset> {
    block_len: usize,
    block_bit_subset_type: PhantomData<S>,
}

impl<S: BlockBitSubset> BlockBitSet<S> {
    pub fn can_fit_multiple_blocks(slab_byte_len: usize, block_byte_len: usize) -> bool {
        1 < block_len_from_slab_byte_len_and_block_byte_len::<S>(slab_byte_len, block_byte_len)
    }

    pub fn len(&self) -> usize {
        self.block_len
    }

    /// Initializes the `BlockBitSet` using the provided pointer and size.
    ///
    /// The pointer is the region where `Self` will be allocated, and `size`
    /// is the amount of space available for blocks, and should already have
    /// accounted for the layout of `Self`. Blocks should be of size `block_size`
    ///
    /// NOTE: This function will write the initial bit vector to memory within
    /// the range represented by the provided pointer and size, so it is absolutely
    /// imperative that the memory was allocated and available
    ///
    /// This function is unsafe for a lot of reasons:
    ///
    /// - If you haven't allocated `size` bytes of memory at `ptr`, you will segfault
    /// - If `ptr` does not point to where `Self` will be written, undefined behavior
    /// - If `ptr` points to something else in use, undefined behavior
    pub unsafe fn write(ptr: *mut Self, size: usize, block_size: usize) {
        let block_len = block_len_from_slab_byte_len_and_block_byte_len::<S>(size, block_size);
        assert!(
            0 < block_len,
            "Slab ({:?} bytes) cannot hold any blocks ({:?} bytes)",
            size,
            block_size
        );
        assert!(
            1 < block_len,
            "Slab ({:?} bytes) can hold only 1 block ({:?} bytes), so it should not use a BlockBitSet and instead use a single-block carrier",
            size,
            block_size
        );

        ptr.write(Self {
            block_len,
            block_bit_subset_type: PhantomData,
        });

        Self::write_subsets(ptr, block_len);
    }

    unsafe fn write_subsets(ptr: *mut Self, block_len: usize) {
        // Calculate pointer to beginning of bit vector
        let subsets_ptr = ptr.add(1) as *mut S;
        let subset_len = subset_len_from_block_len::<S>(block_len);

        // Write initial state to bit vector, using calculated pointer
        for i in 0..subset_len {
            let subset_ptr = subsets_ptr.add(i);
            subset_ptr.write(Default::default());
        }
    }

    unsafe fn subset(&self, index: usize) -> &S {
        let self_ptr = self as *const Self;
        let subsets_ptr = self_ptr.add(1) as *const S;

        &*subsets_ptr.add(index)
    }

    fn subset_bit_len(&self, index: usize) -> usize {
        let subset_bit_size = bit_size_of::<S>();
        let len_before = subset_bit_size * index;
        let remaining_len = self.block_len - len_before;

        if subset_bit_size < remaining_len {
            subset_bit_size
        } else {
            remaining_len
        }
    }

    fn subset_len(&self) -> usize {
        subset_len_from_block_len::<S>(self.block_len)
    }

    fn as_subset_slice(&self) -> &[S] {
        let self_ptr = self as *const Self;

        unsafe {
            let subsets_ptr = self_ptr.add(1) as *const S;

            core::slice::from_raw_parts(subsets_ptr, self.subset_len())
        }
    }

    fn subset_index_and_bit_from_set_bit(set_bit: usize) -> (usize, usize) {
        let subset_bit_size = bit_size_of::<S>();
        let index = set_bit / subset_bit_size;
        let subset_bit = set_bit % subset_bit_size;

        (index, subset_bit)
    }

    fn set_bit_from_subset_index_and_bit(subset_index: usize, subset_bit: usize) -> usize {
        let subset_bit_size = bit_size_of::<S>();

        subset_index * subset_bit_size + subset_bit
    }

    /// Try to allocate block.
    ///
    /// NOTE: This operation can fail, primarily in multi-threaded scenarios
    /// with atomics in play. If `Err(AllocError)` is returned, it means that either all blocks were
    /// already allocated or a neighboring bit that was in the same subset
    /// represented by `S` was flipped. You may retry or simply fail the allocation request. It is
    /// up to the allocator.
    pub fn alloc_block(&self) -> Result<usize, AllocError> {
        for subset_index in 0..self.subset_len() {
            let subset = unsafe { self.subset(subset_index) };
            let subset_bit_len = self.subset_bit_len(subset_index);

            match subset.alloc_block(subset_bit_len) {
                Ok(subset_bit) => {
                    let set_bit = Self::set_bit_from_subset_index_and_bit(subset_index, subset_bit);

                    return Ok(set_bit);
                }
                Err(AllocError) => continue,
            }
        }

        Err(AllocError)
    }

    /// Free the block represented by the given bit index
    pub fn free(&self, bit: usize) {
        let (subset_index, subset_bit) = Self::subset_index_and_bit_from_set_bit(bit);
        let subset = unsafe { self.subset(subset_index) };

        subset.free(subset_bit);
    }

    /// Return a count of the allocated blocks managed by this bit set
    fn count_allocated(&self) -> usize {
        self.as_subset_slice()
            .iter()
            .map(|subset| subset.count_allocated())
            .sum()
    }

    /// Return a count of the free blocks managed by this bit set
    pub fn count_free(&self) -> usize {
        self.block_len - self.count_allocated()
    }

    /// Like `mem::size_of`, but includes the variable size of the subsets
    pub fn size(&self) -> usize {
        Self::size_from_subset_len(self.subset_len())
    }

    fn size_from_subset_len(subset_len: usize) -> usize {
        mem::size_of::<Self>() + subset_len * mem::size_of::<S>()
    }

    #[cfg(test)]
    fn size_from_block_len(block_len: usize) -> usize {
        let subset_len = subset_len_from_block_len::<S>(block_len);

        Self::size_from_subset_len(subset_len)
    }

    /// Returns the number of bytes which would be wasted in a slab
    /// of `slab_byte_len` bytes, containing `Self`, the bit vector, and blocks
    /// of `block_byte_len` bytes. This wastage is due to usable space in
    /// the slab not being evenly divisible by the `block_byte_len`, where
    /// usable space is the space left in the slab after accounting for
    /// the header/bit vector metadata.
    #[cfg(test)]
    fn wastage(slab_byte_len: usize, block_byte_len: usize) -> usize {
        let block_len =
            block_len_from_slab_byte_len_and_block_byte_len::<S>(slab_byte_len, block_byte_len);
        let byte_len = Self::size_from_block_len(block_len);
        let blocks_byte_len = block_len * block_byte_len;

        slab_byte_len - byte_len - blocks_byte_len
    }
}

unsafe impl<S: BlockBitSubset + Sync> Sync for BlockBitSet<S> {}
unsafe impl<S: BlockBitSubset + Send> Send for BlockBitSet<S> {}

/// Calculates the maximum number of blocks which can fit in a region
/// of `slab_byte_len` bytes, where each block is `block_byte_len` bytes.
///
/// NOTE: The slab_byte_len of the bit vector is subtracted from the region,
/// so if you already account for that elsewhere, be sure to increase
/// `slab_byte_len` by that amount.
fn block_len_from_slab_byte_len_and_block_byte_len<S: BlockBitSubset>(
    slab_byte_len: usize,
    block_byte_len: usize,
) -> usize {
    let this = mem::size_of::<BlockBitSet<S>>();
    let max_usable = slab_byte_len - this;
    let subset_block_len = bit_size_of::<S>();
    let bytes_represented_per_unit = mem::size_of::<S>() + (block_byte_len * subset_block_len);

    // The number of usable bytes accounted for so far
    let mut used = 0;
    // The number of blocks we have been able to fit so far
    let mut block_count = 0;
    // While we still have usable space left, calculate how much space is used
    // by a single usize worth of bits and the blocks they track. If the required
    // space fits in the available space, increment the block count and usage
    // accordingly. If the required space cannot fit, then do a final check to see
    // whether a partial usize worth of blocks can fit, and account for those.
    while used < max_usable {
        let new_used = used + bytes_represented_per_unit;
        if new_used > max_usable {
            // We don't have enough room for a full usize worth of blocks
            let available = max_usable - used;
            // Ensure we have at least enough space for the bits themselves
            if available > mem::size_of::<S>() {
                // Account for the number of blocks which can fit in the remaining space
                let available_for_blocks = available - mem::size_of::<S>();
                block_count += num_blocks_fit(available_for_blocks, block_byte_len);
                return block_count;
            }

            break;
        }
        used = new_used;
        block_count += subset_block_len;
    }

    block_count
}

fn subset_len_from_block_len<S: BlockBitSubset>(block_len: usize) -> usize {
    let bits_per_subset = bit_size_of::<S>();
    let subset_len = block_len / bits_per_subset;
    let needs_extra = subset_len % bits_per_subset == 0;

    if needs_extra {
        subset_len + 1
    } else {
        subset_len
    }
}

#[inline(always)]
const fn num_blocks_fit(num_bytes: usize, block_size: usize) -> usize {
    num_bytes / block_size
}

#[cfg(test)]
mod tests {
    use core::mem;

    use intrusive_collections::LinkedListLink;
    use liblumen_alloc_macros::*;
    use liblumen_core::alloc::mmap;
    use liblumen_core::alloc::size_classes::SizeClass;
    use liblumen_core::alloc::Layout;

    use crate::carriers::{self, SlabCarrier};

    use super::*;

    #[derive(SizeClassIndex)]
    struct SizeClassAlloc;

    #[test]
    fn thread_safe_block_bit_set_test() {
        let size = 65536;
        let block_size = mem::size_of::<u64>(); // 8
        let layout = Layout::from_size_align(size, mem::align_of::<AtomicUsize>()).unwrap();

        let block_len = block_len_from_slab_byte_len_and_block_byte_len::<ThreadSafeBlockBitSubset>(
            size, block_size,
        );

        let ptr = unsafe { mmap::map(layout).unwrap() };
        let raw = ptr.as_ptr();
        let block_bit_set_ptr = raw as *mut BlockBitSet<ThreadSafeBlockBitSubset>;

        unsafe {
            BlockBitSet::write(block_bit_set_ptr, size, block_size);
        }

        let block_bit_set = unsafe { &*block_bit_set_ptr };

        assert_eq!(block_bit_set.len(), block_len);
        assert_eq!(block_bit_set.count_free(), block_len);
    }

    #[test]
    fn wastage_for_size_classes_is_acceptable() {
        let size = carriers::SUPERALIGNED_CARRIER_SIZE;
        let acceptable_wastage = 88; // based on the super-aligned size
        let slab_byte_len =
            size - mem::size_of::<SlabCarrier<LinkedListLink, ThreadSafeBlockBitSubset>>();

        for class in SizeClassAlloc::SIZE_CLASSES.iter() {
            let block_byte_len = class.to_bytes();
            let block_word_len = class.as_words();
            let wastage =
                BlockBitSet::<ThreadSafeBlockBitSubset>::wastage(slab_byte_len, block_byte_len);

            assert!(
                wastage <= acceptable_wastage,
                "wastage of {} bytes exceeds acceptable level of {} bytes for size class {} bytes ({} words)",
                wastage,
                acceptable_wastage,
                block_byte_len,
                block_word_len,
            );
        }
    }
}
