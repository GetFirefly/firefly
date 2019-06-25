///! `StandardAlloc` is a general purpose allocator that divides allocations into two major
///! categories: multi-block carriers up to a certain threshold, after which allocations use
///! single-block carriers.
///!
///! Allocations that use multi-block carriers are filled by searching in a balanced
///! binary tree for the first carrier with free blocks of suitable size, then finding the
/// block ! with the best fit within that carrier.
///!
///! Multi-block carriers are allocated using super-aligned boundaries. Specifically, this is
///! 262144 bytes, or put another way, 64 pages that are 4k large. Since that page size is not
///! guaranteed across platforms, it is not of particular significance. However the purpose of
///! super-alignment is to make it trivial to calculate a pointer to the carrier header given
///! a pointer to a block or data within the bounds of that carrier. It is also a convenient
/// size ! as allocations of all size classes both fit into the super-aligned size, and can fit
/// multiple ! blocks in that carrier. For example, the largest size class, 32k, can fit 7
/// blocks of that ! size, and have room remaining for smaller blocks (carrier and block
/// headers take up some space). !
///! Single-block carriers are allocated for any allocation requests above the maximum
/// multi-block ! size class, also called the "single-block carrier threshold". Currently this
/// is statically set ! to the size class already mentioned (32k).
///!
///! The primary difference between the carrier types, other than the size of allocations they
///! handle, is that single-block carriers are always freed, where multi-block carriers are
/// retained ! and reused, the allocator effectively maintaining a cache to more efficiently
/// serve allocations. !
///! The allocator starts with a single multi-block carrier, and additional multi-block
/// carriers are ! allocated as needed when the current carriers are unable to satisfy
/// allocation requests. As ! stated previously, large allocations always allocate in
/// single-block carriers, but none are ! allocated up front.
///!
///! NOTE: It will be important in the future to support carrier migration to allocators in
/// other ! threads to avoid situations where an allocator on one thread is full, so additional
/// carriers ! are allocated when allocators on other threads have carriers that could have
/// filled the request. ! See [CarrierMigration.md] in the OTP documentation for information
/// about how that works and ! the rationale.
use core::alloc::{Alloc, AllocErr, Layout};
use core::cmp;
use core::ptr::{self, NonNull};

#[cfg(not(test))]
use alloc::vec::Vec;

use cfg_if::cfg_if;
use lazy_static::lazy_static;

use intrusive_collections::LinkedListLink;
use intrusive_collections::{Bound, UnsafeRef};
use intrusive_collections::{RBTree, RBTreeLink};

use liblumen_core::alloc::alloc_ref::{self, AsAllocRef};
use liblumen_core::alloc::mmap;
use liblumen_core::locks::SpinLock;
use liblumen_core::util::cache_padded::CachePadded;

use crate::carriers::{superalign_down, SUPERALIGNED_CARRIER_SIZE};
use crate::carriers::{MultiBlockCarrier, SingleBlockCarrier};
use crate::carriers::{MultiBlockCarrierTree, SingleBlockCarrierList};
use crate::sorted::{SortKey, SortOrder, SortedKeyAdapter};
use crate::AllocatorInfo;

// The global instance of StandardAlloc
cfg_if! {
    if #[cfg(feature = "instrument")] {
        use crate::StatsAlloc;
        lazy_static! {
            static ref STD_ALLOC: StatsAlloc<StandardAlloc> = {
                StatsAlloc::new(StandardAlloc::new())
            };
        }
    } else {
        lazy_static! {
            static ref STD_ALLOC: StandardAlloc = StandardAlloc::new();
        }
    }
}

/// Allocates a new block of memory using the given layout
pub unsafe fn alloc(layout: Layout) -> Result<NonNull<u8>, AllocErr> {
    STD_ALLOC.allocate(layout)
}

/// Reallocates a previously allocated block of memory, in-place if possible
pub unsafe fn realloc(
    ptr: NonNull<u8>,
    layout: Layout,
    new_size: usize,
) -> Result<NonNull<u8>, AllocErr> {
    STD_ALLOC.reallocate(ptr, layout, new_size)
}

/// Deallocates a previously allocated block of memory
pub unsafe fn dealloc(ptr: NonNull<u8>, layout: Layout) {
    STD_ALLOC.deallocate(ptr, layout);
}

/// Gets information about the global standard allocator
pub fn alloc_info() -> AllocatorInfo {
    STD_ALLOC.info()
}

struct StandardAlloc {
    sbc_threshold: usize,
    sbc: CachePadded<SpinLock<SingleBlockCarrierList>>,
    mbc: CachePadded<SpinLock<MultiBlockCarrierTree>>,
}
impl StandardAlloc {
    const MAX_SIZE_CLASS: usize = 32 * 1024;

    /// Create a new instance of this allocator
    pub fn new() -> Self {
        // Allocate a default carrier
        // TODO: In the future we may want to do like the BEAM does and
        // have a separate struct field for the main carrier, so that allocations
        // have a fast path if the main carrier has available space
        let main_carrier = unsafe {
            create_multi_block_carrier().expect("unable to allocate main multi-block carrier")
        };
        let mut mbc = RBTree::new(SortedKeyAdapter::new(SortOrder::SizeAddressOrder));
        mbc.insert(main_carrier);

        Self {
            sbc: CachePadded::new(SpinLock::new(SingleBlockCarrierList::default())),
            mbc: CachePadded::new(SpinLock::new(mbc)),
            sbc_threshold: Self::MAX_SIZE_CLASS,
        }
    }

    /// Gets information about this allocator
    pub fn info(&self) -> AllocatorInfo {
        let num_mbc = self.count_mbc();
        let num_sbc = self.count_sbc();
        AllocatorInfo {
            num_multi_block_carriers: num_mbc,
            num_single_block_carriers: num_sbc,
        }
    }

    // Counts the number of multi-block carriers this allocator holds
    fn count_mbc(&self) -> usize {
        let mbc = self.mbc.lock();
        mbc.iter().count()
    }

    // Counts the number of single-block carriers this allocator holds
    fn count_sbc(&self) -> usize {
        let sbc = self.sbc.lock();
        sbc.iter().count()
    }

    unsafe fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocErr> {
        let size = layout.size();
        if size >= self.sbc_threshold {
            return self.alloc_large(layout);
        }

        // From this point onwards, we're working with multi-block carriers
        // First, find a carrier large enough to hold the requested allocation

        // Ensure allocated region has enough space for carrier header and aligned block

        // Start with the first carrier with a usable size of at least `block_size` bytes
        let bound = SortKey::new(SortOrder::SizeAddressOrder, size, 0);
        let mbc = self.mbc.lock();
        let cursor = mbc.lower_bound(Bound::Included(&bound));
        // Try each carrier, from smallest to largest, until we find a fit
        while let Some(carrier) = cursor.get() {
            // In each carrier, try to find a best fit block and allocate it
            if let Some(block) = carrier.alloc_block(&layout) {
                return Ok(block);
            }
        }
        drop(mbc);

        // If we reach this point, no carriers with suitable blocks were available
        // Allocate a new carrier of adequate size and use it to satisfy request
        //
        // Multi-block carriers are currently super-aligned by default, which means
        // we always allocate carriers of the same size, and since the super-aligned size
        // is always larger than the single-block threshold, new multi-block carriers are
        // guaranteed to fulfill the allocation request that caused their creation
        let carrier = create_multi_block_carrier()?;
        let mut mbc = self.mbc.lock();
        mbc.insert(carrier.clone());
        drop(mbc);
        // Allocate block using newly allocated carrier
        // NOTE: It should never be possible for this to fail
        let block = carrier
            .alloc_block(&layout)
            .expect("unexpected block allocation failure");
        // Return data pointer
        Ok(block)
    }

    unsafe fn reallocate(
        &self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<NonNull<u8>, AllocErr> {
        let raw = ptr.as_ptr();
        let size = layout.size();

        if size >= self.sbc_threshold {
            // This was a single-block carrier
            //
            // Even if the new size would fit in a multi-block carrier, we're going
            // to keep the allocation in the single-block carrier, to avoid the extra
            // complexity, as there is little payoff
            return self.realloc_large(ptr, layout, new_size);
        }

        // From this point onwards, we're working with multi-block carriers
        // Locate the owning carrier and try to reallocate using it
        let key = SortKey::new(SortOrder::AddressOrder, 0, raw as usize);
        let mut mbc = self.mbc.lock();
        let cursor = mbc.lower_bound_mut(Bound::Included(&key));
        // This cursor should always return the owning carrier,
        // if it does not, then either the given pointer was allocated
        // using a different allocator, or there is a bug in this implementation
        let carrier = cursor.get().expect("realloc called with invalid pointer");
        let carrier_ptr = superalign_down(raw as usize) as *const u8;
        debug_assert!((carrier as *const _ as *const u8) == carrier_ptr);
        // Attempt reallocation
        if let Some(block) = carrier.realloc_block(raw, &layout, new_size) {
            // We were able to reallocate within this carrier
            return Ok(block);
        }
        drop(mbc);

        // If we reach this point, we have to try allocating a new block and
        // copying the data from the old block to the new one
        let new_layout = Layout::from_size_align(new_size, layout.align()).expect("invalid layout");
        // Allocate new block
        let block = self.allocate(new_layout)?;
        // Copy data from old block into new block
        let blk = block.as_ptr() as *mut u8;
        ptr::copy_nonoverlapping(raw, blk, cmp::min(size, new_size));
        // Free old block
        self.deallocate(ptr, layout);

        // Return new data pointer
        Ok(block)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let ptr = ptr.as_ptr();
        let size = layout.size();

        if size >= self.sbc_threshold {
            // This block would have to be in a single-block carrier
            return self.dealloc_large(ptr);
        }

        // Multi-block carriers are always super-aligned, and no larger
        // than the super-aligned size, so we can find the carrier header
        // trivially using the pointer itself
        let carrier_ptr = superalign_down(ptr as usize) as *const MultiBlockCarrier<RBTreeLink>;
        let carrier = UnsafeRef::from_raw(carrier_ptr);

        // TODO: Perform conditional release of memory back to operating system,
        // for now, we always free single-block carriers, but never multi-block carriers
        let mbc = self.mbc.lock();
        carrier.free_block(ptr, layout);
        drop(mbc)
    }

    /// This function handles allocations which exceed the single-block carrier threshold
    unsafe fn alloc_large(&self, layout: Layout) -> Result<NonNull<u8>, AllocErr> {
        // Ensure allocated region has enough space for carrier header and aligned block
        let data_layout = layout.clone();
        let carrier_layout = Layout::new::<SingleBlockCarrier<LinkedListLink>>();
        let (carrier_layout, data_offset) = carrier_layout.extend(data_layout).unwrap();
        // Track total size for carrier metadata
        let size = carrier_layout.size();
        // Allocate region
        let ptr = mmap::map(carrier_layout)?;
        // Get pointer to carrier header location
        let carrier = ptr.as_ptr() as *mut SingleBlockCarrier<LinkedListLink>;
        // Write initial carrier header
        ptr::write(
            carrier,
            SingleBlockCarrier {
                size,
                layout,
                link: LinkedListLink::new(),
            },
        );
        // Get pointer to data region in allocated carrier+block
        let data = (carrier as *mut u8).offset(data_offset as isize);
        // Cast carrier pointer to UnsafeRef and add to linked list
        // This implicitly mutates the link in the carrier
        let carrier = UnsafeRef::from_raw(carrier);
        let mut sbc = self.sbc.lock();
        sbc.push_front(carrier);
        // Return data pointer
        Ok(NonNull::new_unchecked(data))
    }

    unsafe fn realloc_large(
        &self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<NonNull<u8>, AllocErr> {
        // Allocate new carrier
        let new_ptr =
            self.alloc_large(Layout::from_size_align_unchecked(new_size, layout.align()))?;
        // Copy old data into new carrier
        let old_ptr = ptr.as_ptr();
        let old_size = layout.size();
        ptr::copy_nonoverlapping(old_ptr, new_ptr.as_ptr(), cmp::min(old_size, new_size));
        // Free old carrier
        self.dealloc_large(old_ptr);
        // Return new carrier
        Ok(new_ptr)
    }

    /// This function handles allocations which exceed the single-block carrier threshold
    unsafe fn dealloc_large(&self, ptr: *const u8) {
        // In the case of single-block carriers,
        // we must walk the list until we find the owning carrier
        let mut sbc = self.sbc.lock();
        let mut cursor = sbc.front_mut();
        loop {
            let next = cursor.get();
            debug_assert!(next.is_some(), "invalid free of carrier");

            let carrier = next.unwrap();
            if !carrier.owns(ptr) {
                cursor.move_next();
                continue;
            }

            let carrier_ptr = carrier as *const _ as *mut u8;

            // Calculate the layout of the allocated carrier
            //   - First, get layout of carrier header
            //   - Extend the layout with the block layout to get the original layout used in
            //     `try_alloc`
            let (layout, _) = Layout::new::<SingleBlockCarrier<LinkedListLink>>()
                .extend(carrier.layout())
                .unwrap();
            // Unlink the carrier from the linked list
            let _ = cursor.remove();
            // Release memory for carrier to OS
            mmap::unmap(carrier_ptr, layout);

            return;
        }
    }
}
unsafe impl Alloc for StandardAlloc {
    #[inline]
    unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocErr> {
        self.allocate(layout)
    }

    #[inline]
    unsafe fn realloc(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<NonNull<u8>, AllocErr> {
        self.reallocate(ptr, layout, new_size)
    }

    #[inline]
    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        self.deallocate(ptr, layout)
    }
}
impl<'a> AsAllocRef<'a> for StandardAlloc {
    type Handle = alloc_ref::Handle<'a, Self>;

    #[inline]
    fn as_alloc_ref(&self) -> Self::Handle {
        alloc_ref::Handle::new(self)
    }
}
impl Drop for StandardAlloc {
    fn drop(&mut self) {
        // Drop single-block carriers
        let mut sbc = self.sbc.lock();
        // We have to dynamically allocate this vec because the only
        // other method we have of collecting the pointer/layout combo
        // is by using a cursor, and the current API is not flexible enough
        // to allow us to walk the tree freeing each element after we've moved
        // the cursor past it. Instead we gather all the pointers to clean up
        // and do it all at once at the end
        //
        // NOTE: May be worth exploring a `Drop` impl for the carriers
        let mut carriers = sbc
            .iter()
            .map(|carrier| (carrier as *const _ as *mut _, carrier.layout()))
            .collect::<Vec<_>>();

        // Prevent the list from trying to drop memory that has now been freed
        sbc.fast_clear();

        // Actually drop the carriers
        for (ptr, layout) in carriers.drain(..) {
            unsafe {
                mmap::unmap(ptr, layout);
            }
        }

        // Drop multi-block carriers
        let mbc_size = SUPERALIGNED_CARRIER_SIZE;
        let mbc_layout = unsafe { Layout::from_size_align_unchecked(mbc_size, mbc_size) };
        let mut mbc = self.mbc.lock();
        let mut carriers = mbc
            .iter()
            .map(|carrier| (carrier as *const _ as *mut _, mbc_layout.clone()))
            .collect::<Vec<_>>();

        // Prevent the tree from trying to drop memory that has now been freed
        mbc.fast_clear();

        for (ptr, layout) in carriers.drain(..) {
            unsafe {
                mmap::unmap(ptr, layout);
            }
        }
    }
}
unsafe impl Sync for StandardAlloc {}
unsafe impl Send for StandardAlloc {}

/// Creates a new, empty multi-block carrier, unlinked to the allocator
///
/// The carrier is allocated via mmap on supported platforms, or the system
/// allocator otherwise.
///
/// NOTE: You must make sure to add the carrier to the free list of the
/// allocator, or it will not be used, and will not be freed
unsafe fn create_multi_block_carrier() -> Result<UnsafeRef<MultiBlockCarrier<RBTreeLink>>, AllocErr>
{
    let size = SUPERALIGNED_CARRIER_SIZE;
    let carrier_layout = Layout::from_size_align_unchecked(size, size);
    // Allocate raw memory for carrier
    let ptr = mmap::map(carrier_layout)?;
    // Initialize carrier in memory
    let carrier = MultiBlockCarrier::init(ptr, size);
    // Return an unsafe ref to this carrier back to the caller
    Ok(UnsafeRef::from_raw(carrier))
}

#[cfg(test)]
mod tests {
    use super::*;

    use liblumen_core::alloc::boxed::Box;
    use liblumen_core::alloc::vec::Vec;

    #[test]
    fn std_alloc_small_test() {
        let allocator = StandardAlloc::new();

        // Allocate an object on the heap
        let foo = Box::from_str("just a test", &allocator);

        // Drop the boxed string
        drop(foo);

        assert!(true);
    }

    #[test]
    fn std_alloc_large_test() {
        let allocator = StandardAlloc::new();

        // Allocate a large object on the heap
        let mut foo = Vec::with_capacity(StandardAlloc::MAX_SIZE_CLASS + 1, &allocator);
        foo.push(1u8);

        // Drop it
        drop(foo);

        assert!(true);
    }
}
