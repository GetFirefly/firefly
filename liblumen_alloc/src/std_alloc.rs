use core::alloc::{Alloc, AllocErr, Layout};
///! This module provides a general purpose allocator for use with
///! the Erlang Runtime System. Specifically it is optimized for
///! general usage, where allocation patterns are unpredictable, or
///! for allocations where a more specialized allocator is unsuitable
///! or unavailable.
use core::ptr::{self, NonNull};

use intrusive_collections::{intrusive_adapter, Bound, UnsafeRef};
use intrusive_collections::{LinkedList, LinkedListLink};
use intrusive_collections::{RBTree, RBTreeLink};

use crate::mmap;
//use crate::size_classes;
use crate::block::{Block, FreeBlockTree};
use crate::carriers::{MultiBlockCarrier, SingleBlockCarrier};
use crate::erts::SpinLock;
use crate::sorted::{SortKey, SortOrder, SortedKeyAdapter};

// Type alias for the list of currently allocated single-block carriers
type SingleBlockCarrierList = LinkedList<SingleBlockCarrierListAdapter>;
// Type alias for the ordered tree of currently allocated multi-block carriers
type MultiBlockCarrierTree = RBTree<SortedKeyAdapter<MultiBlockCarrier<RBTreeLink>>>;

// Implementation of adapter for intrusive collection used for single-block carriers
intrusive_adapter!(SingleBlockCarrierListAdapter = UnsafeRef<SingleBlockCarrier<LinkedListLink>>: SingleBlockCarrier<LinkedListLink> { link: LinkedListLink });

/// `StandardAlloc` is a general purpose allocator that divides allocations into two major
/// categories: multi-block carriers up to a certain threshold, after which allocations use
/// single-block carriers.
///
/// Allocations that use multi-block carriers are filled by searching in a balanced
/// binary tree for the first carrier with free blocks of suitable size, then finding the block
/// with the best fit within that carrier.
///
/// Multi-block carriers are allocated using super-aligned boundaries. Specifically, this is
/// 262144 bytes, or put another way, 64 pages that are 4k large. Since that page size is not
/// guaranteed across platforms, it is not of particular significance. However the purpose of
/// super-alignment is to make it trivial to calculate a pointer to the carrier header given
/// a pointer to a block or data within the bounds of that carrier. It is also a convenient size
/// as allocations of all size classes both fit into the super-aligned size, and can fit multiple
/// blocks in that carrier. For example, the largest size class, 32k, can fit 7 blocks of that
/// size, and have room remaining for smaller blocks (carrier and block headers take up some space).
///
/// Single-block carriers are allocated for any allocation requests above the maximum multi-block
/// size class, also called the "single-block carrier threshold". Currently this is statically set
/// to the size class already mentioned (32k).
///
/// The primary difference between the carrier types, other than the size of allocations they
/// handle, is that single-block carriers are always freed, where multi-block carriers are retained
/// and reused, the allocator effectively maintaining a cache to more efficiently serve allocations.
///
/// The allocator starts with a single multi-block carrier, and additional multi-block carriers are
/// allocated as needed when the current carriers are unable to satisfy allocation requests. As
/// stated previously, large allocations always allocate in single-block carriers, but none are
/// allocated up front.
///
/// NOTE: It will be important in the future to support carrier migration to allocators in other
/// threads to avoid situations where an allocator on one thread is full, so additional carriers
/// are allocated when allocators on other threads have carriers that could have filled the request.
/// See [CarrierMigration.md] in the OTP documentation for information about how that works and
/// the rationale.
pub struct StandardAlloc {
    sbc_threshold: usize,
    sbc: SpinLock<SingleBlockCarrierList>,
    mbc: SpinLock<MultiBlockCarrierTree>,
}
impl StandardAlloc {
    // The number of bits to shift/mask to find a superaligned address
    const SA_BITS: usize = 18;
    // The number of bits to shift to find a superaligned carrier address
    const SA_CARRIER_SHIFT: usize = Self::SA_BITS;
    // The size of a superaligned carrier, 262k (262,144 bytes)
    const SA_CARRIER_SIZE: usize = 1usize << Self::SA_CARRIER_SHIFT;
    // The mask needed to go from a pointer in a SA carrier to the carrier
    const SA_CARRIER_MASK: usize = (!0usize) << Self::SA_CARRIER_SHIFT;

    // TODO: Switch this back to the constant in `size_classes` when that module is done
    const MAX_SIZE_CLASS: usize = 32 * 1024;

    pub fn new() -> Self {
        Self {
            sbc: SpinLock::new(LinkedList::new(SingleBlockCarrierListAdapter::new())),
            mbc: SpinLock::new(RBTree::new(SortedKeyAdapter::new(
                SortOrder::SizeAddressOrder,
            ))),
            sbc_threshold: Self::MAX_SIZE_CLASS,
        }
    }
}

unsafe impl Alloc for StandardAlloc {
    unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocErr> {
        let size = layout.size();
        if size >= self.sbc_threshold {
            return self.alloc_large(layout);
        }

        // From this point onwards, we're working with multi-block carriers
        // First, find a carrier large enough to hold the requested allocation

        // Ensure allocated region has enough space for carrier header and aligned block
        let (block_layout, _data_offset) = Layout::new::<Block>().extend(layout.clone()).unwrap();
        let block_size = block_layout.size();

        // Start with the first carrier with a usable size of at least `size` bytes
        let bound = SortKey::new(SortOrder::SizeAddressOrder, block_size, 0);
        let mbc = self.mbc.lock();
        let cursor = mbc.lower_bound(Bound::Included(&bound));
        // Try each carrier, from smallest to largest, until we find a fit
        while let Some(carrier) = cursor.get() {
            // In each carrier, try to find a best fit block and allocate it
            if let Some(block) = carrier.alloc_block(&block_layout) {
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

        let size = Self::SA_CARRIER_SIZE;
        let carrier_layout = Layout::from_size_align_unchecked(size, size);
        // Allocate region
        let ptr = mmap::map(carrier_layout)?;
        // Get pointer to carrier header location
        let carrier = ptr.as_ptr() as *mut MultiBlockCarrier<RBTreeLink>;
        // Write initial carrier header
        ptr::write(
            carrier,
            MultiBlockCarrier {
                size,
                link: RBTreeLink::new(),
                blocks: FreeBlockTree::new(SortOrder::SizeAddressOrder),
            },
        );
        // Cast carrier pointer to UnsafeRef and add to multi-block carrier tree
        // This implicitly mutates the link in the carrier
        let carrier = UnsafeRef::from_raw(carrier);
        let mut mbc = self.mbc.lock();
        mbc.insert(carrier.clone());
        drop(mbc);
        // Allocate block using newly allocated carrier
        // NOTE: It should never be possible for this to fail
        let block = carrier.alloc_block(&layout);
        assert!(block.is_some(), "block allocation unexpectedly failed");
        // Return data pointer
        Ok(block.unwrap())
    }

    unsafe fn realloc(
        &mut self,
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
        // TODO: Carrier needs to be mutable here
        let carrier_ptr = Self::superaligned_floor(raw as usize) as *const u8;
        debug_assert!((carrier as *const _ as *const u8) == carrier_ptr);
        // Attempt reallocation
        if let Some(block) = carrier.realloc_block(raw, &layout, new_size) {
            // We were able to reallocate within this carrier
            return Ok(block);
        }
        drop(mbc);
        // If we reach this point, we have to try allocating a new block and
        // copying the data from the old block to the new one
        let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
        // Allocate new block
        let block = self.alloc(new_layout)?;
        // Copy data from old block into new block
        let blk = block.as_ptr() as *mut u8;
        ptr::copy_nonoverlapping(raw, blk, layout.size());
        // Free old block
        self.dealloc(ptr, layout);
        // Return new data pointer
        Ok(block)
    }

    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        let ptr = ptr.as_ptr();
        let size = layout.size();

        if size >= self.sbc_threshold {
            // This block would have to be in a single-block carrier
            return self.dealloc_large(ptr);
        }

        // Multi-block carriers are always super-aligned, and no larger
        // than the super-aligned size, so we can find the carrier header
        // trivially using the pointer itself
        let carrier_ptr =
            Self::superaligned_floor(ptr as usize) as *const MultiBlockCarrier<RBTreeLink>;
        let carrier = UnsafeRef::from_raw(carrier_ptr);

        // TODO: Perform conditional release of memory back to operating system,
        // for now, we always free single-block carriers, but never multi-block carriers
        carrier.free_block(ptr, layout);
    }
}

impl StandardAlloc {
    /// This function handles allocations which exceed the single-block carrier threshold
    unsafe fn alloc_large(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocErr> {
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
        &mut self,
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
        ptr::copy_nonoverlapping(old_ptr, new_ptr.as_ptr(), old_size);
        // Free old carrier
        self.dealloc_large(old_ptr);
        // Return new carrier
        Ok(new_ptr)
    }

    /// This function handles allocations which exceed the single-block carrier threshold
    unsafe fn dealloc_large(&mut self, ptr: *const u8) {
        // In the case of single-block carriers,
        // we must walk the list until we find the owning carrier
        let mut sbc = self.sbc.lock();
        let mut cursor = sbc.front_mut();
        loop {
            let next = cursor.get();
            debug_assert!(next.is_some(), "invalid free of carrier");

            let sbc = next.unwrap();
            if !sbc.owns(ptr) {
                cursor.move_next();
                continue;
            }

            let carrier_ptr = sbc as *const _ as *mut u8;

            // Calculate the layout of the allocated carrier
            //   - First, get layout of carrier header
            //   - Extend the layout with the block layout to get the original layout used in
            //     `try_alloc`
            let (layout, _) = Layout::new::<SingleBlockCarrier<LinkedListLink>>()
                .extend(sbc.layout())
                .unwrap();
            // Unlink the carrier from the linked list
            let _ = cursor.remove();
            // Release memory for carrier to OS
            mmap::unmap(carrier_ptr, layout);

            return;
        }
    }

    #[inline(always)]
    fn superaligned_floor(addr: usize) -> usize {
        addr & Self::SA_CARRIER_MASK
    }

    #[allow(unused)]
    #[inline(always)]
    fn superaligned_ceil(addr: usize) -> usize {
        Self::superaligned_floor(addr + !Self::SA_CARRIER_MASK)
    }
}
