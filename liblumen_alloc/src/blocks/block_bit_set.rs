use core::alloc::Layout;
use core::mem;
use core::ptr;
use core::sync::atomic::{AtomicUsize, Ordering};

/// This trait abstracts out the concept of a bit set which tracks
/// free/allocated blocks within some contiguous region of memory.
///
/// The actual implementation may differ based on whether it uses atomics
/// or not, and what granularity is used for the byte representation, i.e.
/// u8, u16, usize, etc.
pub trait BlockBitSet: Sized {
    type Repr;

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
    unsafe fn new(ptr: *mut Self, size: usize, block_size: usize) -> Self;
    /// Gets the number of elements in this bit vector
    fn size(&self) -> usize;
    /// Gets the number of bytes occupied by this struct in memory,
    /// not counting `mem::size_of::<Self>()`
    fn extent_size(&self) -> usize;
    /// Determine if the block at the given bit index is allocated
    fn is_allocated(&self, bit: usize) -> bool;
    /// Try and allocate the block represented by the given bit index.
    ///
    /// NOTE: This operation can fail, primarily in multi-threaded scenarios
    /// with atomics in play. If false is returned, it means that either the block was
    /// already allocated in the span of time between when you looked up the block
    /// and when you tried to allocate it, or a neighboring bit that was in the same unit
    /// represented by `Self::Repr` was flipped. You may retry, or try searching for another
    /// block, or simply fail the allocation request. It is up to the allocator.
    fn try_alloc(&self, bit: usize) -> bool;
    /// Free the block represented by the given bit index
    fn free(&self, bit: usize);
    /// Return a count of the free blocks managed by this bit set
    fn count_free(&self) -> usize;
    /// Return a count of the allocated blocks managed by this bit set
    fn count_allocated(&self) -> usize;
    /// Create an iterator from this bit set for iterating over the bits
    /// for every block represented in the bit set
    fn iter(&self) -> BlockBitSetIter<Self>;

    /// Returns the number of bytes which would be wasted in a slab
    /// of `size` bytes, containing `Self`, the bit vector, and blocks
    /// of `block_size` bytes. This wastage is due to usable space in
    /// the slab not being evenly divisible by the block size, where
    /// usable space is the space left in the slab after accounting for
    /// the header/bit vector metadata.
    fn wastage(size: usize, block_size: usize) -> usize;

    /// Calculates the memory layout which extends the layout of `Self` and
    /// is used to hold the bits needed for the underlying bit vector it manages
    #[inline]
    fn extended_layout(num_blocks: usize) -> Layout {
        let align = mem::align_of::<Self::Repr>();
        let required = units_required_for::<Self>(num_blocks) * mem::size_of::<Self::Repr>();
        unsafe { Layout::from_size_align_unchecked(required, align) }
    }
}

/// An implementation of `BlockBitSet` which uses atomics to ensure thread safety
pub struct ThreadSafeBlockBitSet {
    size: usize,
    extent_size: usize,
    vector: *mut AtomicUsize,
}
impl ThreadSafeBlockBitSet {
    #[inline(always)]
    fn get_element<'a>(&self, index: usize) -> &'a mut AtomicUsize {
        unsafe { &mut *(self.vector.offset(index as isize)) }
    }

    #[inline(always)]
    fn get_element_for_bit<'a>(&self, bit: usize) -> &'a mut AtomicUsize {
        self.get_element(bit / mem::size_of::<usize>())
    }
}

impl BlockBitSet for ThreadSafeBlockBitSet {
    type Repr = AtomicUsize;

    #[inline]
    unsafe fn new(ptr: *mut Self, size: usize, block_size: usize) -> Self {
        let num_blocks = calculate_block_fit::<Self>(size, block_size);
        let extent_layout = Self::extended_layout(num_blocks);
        // Calculate pointer to beginning of bit vector
        let vector = ptr.offset(mem::size_of::<Self>() as isize) as *mut AtomicUsize;
        // Write initial state to bit vector, using calculated pointer
        let num_elems = units_required_for::<Self>(num_blocks);
        for i in 0..num_elems {
            let elem = vector.offset(i as isize);
            ptr::write(elem, AtomicUsize::new(0))
        }
        Self {
            size: num_blocks,
            extent_size: extent_layout.size(),
            vector,
        }
    }

    #[inline]
    fn size(&self) -> usize {
        self.size
    }

    #[inline]
    fn extent_size(&self) -> usize {
        self.extent_size
    }

    #[inline]
    fn wastage(size: usize, block_size: usize) -> usize {
        let num_blocks = calculate_block_fit::<Self>(size, block_size);
        let meta_layout = Self::extended_layout(num_blocks);
        let block_bytes = num_blocks * block_size;
        size - meta_layout.size() - block_bytes
    }

    #[inline]
    fn is_allocated(&self, bit: usize) -> bool {
        let shift = bit % mem::size_of::<usize>();
        let flag = 1usize << shift;
        let elem = self.get_element_for_bit(bit);
        elem.load(Ordering::Acquire) & flag == flag
    }

    #[inline]
    fn try_alloc(&self, bit: usize) -> bool {
        let shift = bit % mem::size_of::<usize>();
        let flag = 1usize << shift;
        let elem = self.get_element_for_bit(bit);
        let current = elem.load(Ordering::Acquire);
        if current & flag == flag {
            // Already allocated
            return false;
        }
        elem.compare_exchange(current, current | flag, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    #[inline]
    fn free(&self, bit: usize) {
        let shift = bit % mem::size_of::<usize>();
        let flag = 1usize << shift;
        let elem = self.get_element_for_bit(bit);
        elem.fetch_and(!flag, Ordering::AcqRel);
    }

    #[inline]
    fn count_free(&self) -> usize {
        let mut num_blocks = self.size;
        let num_elems = units_required_for::<Self>(num_blocks);
        for i in 0..num_elems {
            let elem = self.get_element(i);
            let val = elem.load(Ordering::Acquire);
            let allocated = val.count_ones();
            num_blocks -= allocated as usize;
        }
        num_blocks
    }

    #[inline]
    fn count_allocated(&self) -> usize {
        let mut num_blocks = 0;
        let num_elems = units_required_for::<Self>(self.size);
        for i in 0..num_elems {
            let elem = self.get_element(i);
            let val = elem.load(Ordering::Acquire);
            num_blocks += val.count_ones();
        }
        num_blocks as usize
    }

    #[inline]
    fn iter(&self) -> BlockBitSetIter<Self> {
        BlockBitSetIter {
            vector: Self {
                size: self.size,
                extent_size: self.extent_size,
                vector: self.vector,
            },
            bit: 0,
        }
    }
}
unsafe impl Sync for ThreadSafeBlockBitSet {}
unsafe impl Send for ThreadSafeBlockBitSet {}

/// An implementation of `BlockBitSet` which is designed for single-threaded use
pub struct ThreadLocalBlockBitSet {
    size: usize,
    extent_size: usize,
    vector: *mut usize,
}
impl ThreadLocalBlockBitSet {
    #[inline(always)]
    fn get_element<'a>(&self, index: usize) -> &'a mut usize {
        unsafe { &mut *(self.vector.offset(index as isize)) }
    }

    #[inline(always)]
    fn get_element_for_bit<'a>(&self, bit: usize) -> &'a mut usize {
        self.get_element(bit / mem::size_of::<usize>())
    }
}

impl BlockBitSet for ThreadLocalBlockBitSet {
    type Repr = usize;

    #[inline]
    unsafe fn new(ptr: *mut Self, size: usize, block_size: usize) -> Self {
        let num_blocks = calculate_block_fit::<Self>(size, block_size);
        let extent_layout = Self::extended_layout(num_blocks);
        // Calculate pointer to beginning of bit vector
        let vector = ptr.offset(mem::size_of::<Self>() as isize) as *mut usize;
        // Write initial state to bit vector, using calculated pointer
        let num_elems = units_required_for::<Self>(num_blocks);
        for i in 0..num_elems {
            let elem = vector.offset(i as isize);
            ptr::write(elem, 0)
        }
        Self {
            size: num_blocks,
            extent_size: extent_layout.size(),
            vector,
        }
    }

    #[inline]
    fn size(&self) -> usize {
        self.size
    }

    #[inline]
    fn extent_size(&self) -> usize {
        self.extent_size
    }

    #[inline]
    fn wastage(size: usize, block_size: usize) -> usize {
        let num_blocks = calculate_block_fit::<Self>(size, block_size);
        let meta_layout = Self::extended_layout(num_blocks);
        let block_bytes = num_blocks * block_size;
        size - meta_layout.size() - block_bytes
    }

    #[inline]
    fn is_allocated(&self, bit: usize) -> bool {
        let shift = bit % mem::size_of::<usize>();
        let flag = 1usize << shift;
        let elem = self.get_element_for_bit(bit);
        *elem & flag == flag
    }

    #[inline]
    fn try_alloc(&self, bit: usize) -> bool {
        let shift = bit % mem::size_of::<usize>();
        let flag = 1usize << shift;
        let elem = self.get_element_for_bit(bit);
        if *elem & flag == flag {
            // Already allocated
            return false;
        }
        *elem |= flag;
        true
    }

    #[inline]
    fn free(&self, bit: usize) {
        let shift = bit % mem::size_of::<usize>();
        let flag = 1usize << shift;
        let elem = self.get_element_for_bit(bit);
        *elem &= !flag;
    }

    #[inline]
    fn count_free(&self) -> usize {
        let mut num_blocks = self.size;
        let num_elems = units_required_for::<Self>(num_blocks);
        for i in 0..num_elems {
            let elem = self.get_element(i);
            let allocated = elem.count_ones();
            num_blocks -= allocated as usize;
        }
        num_blocks
    }

    #[inline]
    fn count_allocated(&self) -> usize {
        let mut num_blocks = 0;
        let num_elems = units_required_for::<Self>(self.size);
        for i in 0..num_elems {
            let elem = self.get_element(i);
            num_blocks += elem.count_ones();
        }
        num_blocks as usize
    }

    #[inline]
    fn iter(&self) -> BlockBitSetIter<Self> {
        BlockBitSetIter {
            vector: Self {
                size: self.size,
                extent_size: self.extent_size,
                vector: self.vector,
            },
            bit: 0,
        }
    }
}

/// An iterator specifically for `BlockBitSet` implementations,
pub struct BlockBitSetIter<T: BlockBitSet> {
    vector: T,
    bit: usize,
}
impl<B: BlockBitSet> Iterator for BlockBitSetIter<B> {
    type Item = bool;

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.vector.size(), Some(self.vector.size()))
    }

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.bit == self.vector.size() {
            return None;
        }
        let bit = self.bit;
        self.bit += 1;
        Some(self.vector.is_allocated(bit))
    }
}

/// Calculates the maximum number of blocks which can fit in a region
/// of `size` bytes, where each block is `block_size` bytes.
///
/// NOTE: The size of the bit vector is subtracted from the region,
/// so if you already account for that elsewhere, be sure to increase
/// `size` by that amount.
#[inline]
fn calculate_block_fit<T: BlockBitSet>(size: usize, block_size: usize) -> usize {
    let this = mem::size_of::<T>();
    let max_usable = size - this;
    let blocks_per_unit = mem::size_of::<T::Repr>() * 8;
    let bytes_represented_per_unit = mem::size_of::<T::Repr>() + (block_size * blocks_per_unit);

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
            if available > mem::size_of::<T::Repr>() {
                // Account for the number of blocks which can fit in the remaining space
                let available_for_blocks = available - mem::size_of::<T::Repr>();
                block_count += num_blocks_fit(available_for_blocks, block_size);
                return block_count;
            }

            break;
        }
        used = new_used;
        block_count += blocks_per_unit;
    }

    block_count
}

#[inline]
fn units_required_for<T: BlockBitSet>(num_blocks: usize) -> usize {
    let bits_per_unit = mem::size_of::<T::Repr>() * 8;
    let count = num_blocks / bits_per_unit;
    let needs_extra = count % bits_per_unit == 0;
    if needs_extra {
        count + 1
    } else {
        count
    }
}

#[inline(always)]
const fn num_blocks_fit(num_bytes: usize, block_size: usize) -> usize {
    num_bytes / block_size
}

#[cfg(test)]
mod tests {
    use core::alloc::Layout;
    use core::mem;
    use core::ptr;

    use intrusive_collections::LinkedListLink;
    use liblumen_alloc_macros::*;
    use liblumen_core::alloc::mmap;
    use liblumen_core::alloc::size_classes::SizeClass;

    use crate::carriers::{self, SlabCarrier};

    use super::*;

    #[derive(SizeClassIndex)]
    struct SizeClassAlloc;

    #[test]
    fn thread_safe_block_set_test() {
        let size = 65536;
        let block_size = mem::size_of::<u64>(); // 8
        let layout = Layout::from_size_align(size, mem::align_of::<AtomicUsize>()).unwrap();

        let num_blocks = calculate_block_fit::<ThreadSafeBlockBitSet>(size, block_size);

        let ptr = unsafe { mmap::map(layout).unwrap() };
        let raw = ptr.as_ptr();
        let header_ptr = raw as *mut ThreadSafeBlockBitSet;
        unsafe {
            ptr::write(
                header_ptr,
                ThreadSafeBlockBitSet::new(header_ptr, size, block_size),
            );
        }

        let blocks = unsafe { &*header_ptr };

        assert_eq!(blocks.size(), num_blocks);
        assert_eq!(blocks.count_free(), num_blocks);
    }

    #[test]
    fn wastage_for_size_classes_is_acceptable() {
        let size = carriers::SUPERALIGNED_CARRIER_SIZE;
        let acceptable_wastage = 88; // based on the super-aligned size
        let usable_size = (size
            - mem::size_of::<SlabCarrier<LinkedListLink, ThreadSafeBlockBitSet>>())
            + mem::size_of::<ThreadSafeBlockBitSet>();
        for class in SizeClassAlloc::SIZE_CLASSES.iter() {
            let block_size = class.to_bytes();
            let word_size = class.as_words();
            let wastage = ThreadSafeBlockBitSet::wastage(usable_size, block_size);
            assert!(
                wastage <= acceptable_wastage,
                "wastage of {} bytes exceeds acceptable level of {} bytes for size class {} bytes ({} words)", 
                wastage,
                acceptable_wastage,
                block_size,
                word_size,
            );
        }
    }
}
