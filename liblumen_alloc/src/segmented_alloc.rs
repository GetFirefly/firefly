use core::cmp;
use core::intrinsics::unlikely;
use core::ptr::{self, NonNull};

#[cfg(not(test))]
use alloc::boxed::Box;
#[cfg(not(test))]
use alloc::vec::Vec;

use intrusive_collections::{LinkedListLink, UnsafeRef};

use liblumen_alloc_macros::*;
use liblumen_core::alloc::mmap;
use liblumen_core::alloc::prelude::*;
use liblumen_core::alloc::size_classes::{SizeClass, SizeClassIndex};
use liblumen_core::locks::RwLock;

use crate::blocks::ThreadSafeBlockBitSubset;
use crate::carriers::{superalign_down, SUPERALIGNED_CARRIER_SIZE};
use crate::carriers::{SingleBlockCarrier, SlabCarrier};
use crate::carriers::{SingleBlockCarrierList, SlabCarrierList};

/// Like `StandardAlloc`, `SegmentedAlloc` splits allocations into two major categories,
/// multi-block carriers up to a certain threshold, after which allocations use single-block
/// carriers.
///
/// However, `SegmentedAlloc` differs in some key ways:
///
/// - The multi-block carriers are allocated into multiple size classes, where each carrier belongs
///   to a size class and therefore only fulfills allocation requests for blocks of uniform size.
/// - The single-block carrier threshold is statically determined based on the maximum size class
///   for the multi-block carriers and is therefore not configurable.
///
/// Each size class for multi-block carriers contains at least one carrier, and new carriers are
/// allocated as needed when the carrier(s) for a size class are unable to fulfill allocation
/// requests.
///
/// Allocations of blocks in multi-block carriers are filled using address order for both carriers
/// and blocks to reduce fragmentation and improve allocation locality for allocations that fall
/// within the same size class.
#[derive(SizeClassIndex)]
pub struct SegmentedAlloc {
    sbc_threshold: usize,
    sbc: RwLock<SingleBlockCarrierList>,
    classes: Box<[RwLock<SlabCarrierList>]>,
}
impl SegmentedAlloc {
    /// Create a new instance of this allocator
    pub fn new() -> Self {
        // Initialize to default set of empty slab lists
        let mut classes = Vec::with_capacity(Self::NUM_SIZE_CLASSES);
        // Initialize every size class with a single slab carrier
        for size_class in Self::SIZE_CLASSES.iter() {
            let mut list = SlabCarrierList::default();
            let slab = unsafe { Self::create_slab_carrier(*size_class).unwrap() };
            list.push_front(unsafe { UnsafeRef::from_raw(slab) });
            classes.push(RwLock::new(list));
        }
        Self {
            sbc_threshold: Self::MAX_SIZE_CLASS.to_bytes(),
            sbc: RwLock::default(),
            classes: classes.into_boxed_slice(),
        }
    }

    /// Creates a new, empty slab carrier, unlinked to the allocator
    ///
    /// The carrier is allocated via mmap on supported platforms, or the system
    /// allocator otherwise.
    ///
    /// NOTE: You must make sure to add the carrier to the free list of the
    /// allocator, or it will not be used, and will not be freed
    unsafe fn create_carrier(
        size_class: SizeClass,
    ) -> Result<*mut SlabCarrier<LinkedListLink, ThreadSafeBlockBitSubset>, AllocError> {
        let size = SUPERALIGNED_CARRIER_SIZE;
        assert!(size_class.to_bytes() < size);
        let carrier_layout = Layout::from_size_align_unchecked(size, size);
        // Allocate raw memory for carrier
        let ptr = mmap::map(carrier_layout)?;
        // Initialize carrier in memory
        let carrier = SlabCarrier::init(ptr.as_ptr(), size, size_class);
        // Return an unsafe ref to this carrier back to the caller
        Ok(carrier)
    }
}

unsafe impl Allocator for SegmentedAlloc {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() >= self.sbc_threshold {
            return unsafe { self.alloc_large(layout) };
        }
        unsafe { self.alloc_sized(layout) }
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() >= self.sbc_threshold {
            // This block would have to be in a single-block carrier
            return self.dealloc_large(ptr.as_ptr());
        }

        self.dealloc_sized(ptr, layout);
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if old_layout.size() >= self.sbc_threshold {
            // This was a single-block carrier
            //
            // Even if the new size would fit in a multi-block carrier, we're going
            // to keep the allocation in the single-block carrier, to avoid the extra
            // complexity, as there is little payoff
            return self.realloc_large(ptr, old_layout, new_layout);
        }

        self.realloc_sized(ptr, old_layout, new_layout)
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if old_layout.size() >= self.sbc_threshold {
            // This was a single-block carrier
            //
            // Even if the new size would fit in a multi-block carrier, we're going
            // to keep the allocation in the single-block carrier, to avoid the extra
            // complexity, as there is little payoff
            return self.realloc_large(ptr, old_layout, new_layout);
        }

        self.realloc_sized(ptr, old_layout, new_layout)
    }
}

impl SegmentedAlloc {
    /// This function handles allocations which exceed the single-block carrier threshold
    unsafe fn alloc_large(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        // Ensure allocated region has enough space for carrier header and aligned block
        let data_layout = layout.clone();
        let data_layout_size = data_layout.size();
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
        let data = (carrier as *mut u8).add(data_offset);
        // Cast carrier pointer to UnsafeRef and add to linked list
        // This implicitly mutates the link in the carrier
        let carrier = UnsafeRef::from_raw(carrier);
        let mut sbc = self.sbc.write();
        sbc.push_front(carrier);

        let non_null_byte_slice =
            NonNull::slice_from_raw_parts(NonNull::new_unchecked(data), data_layout_size);

        // Return data pointer
        Ok(non_null_byte_slice)
    }

    unsafe fn realloc_large(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // Allocate new carrier
        let non_null_byte_slice = self.alloc_large(new_layout)?;
        // Copy old data into new carrier
        let old_ptr = ptr.as_ptr();
        let old_size = old_layout.size();
        ptr::copy_nonoverlapping(
            old_ptr,
            non_null_byte_slice.as_mut_ptr(),
            cmp::min(old_size, non_null_byte_slice.len()),
        );
        // Free old carrier
        self.dealloc_large(old_ptr);
        // Return new carrier
        Ok(non_null_byte_slice)
    }

    /// This function handles allocations which exceed the single-block carrier threshold
    unsafe fn dealloc_large(&self, ptr: *const u8) {
        // In the case of single-block carriers,
        // we must walk the list until we find the owning carrier
        let mut sbc = self.sbc.write();
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

impl SegmentedAlloc {
    /// Creates a new, empty slab carrier, unlinked to the allocator
    ///
    /// The carrier is allocated via mmap on supported platforms, or the system
    /// allocator otherwise.
    ///
    /// NOTE: You must make sure to add the carrier to the free list of the
    /// allocator, or it will not be used, and will not be freed
    unsafe fn create_slab_carrier(
        size_class: SizeClass,
    ) -> Result<*mut SlabCarrier<LinkedListLink, ThreadSafeBlockBitSubset>, AllocError> {
        let size = SUPERALIGNED_CARRIER_SIZE;
        assert!(size_class.to_bytes() < size);
        let carrier_layout = Layout::from_size_align_unchecked(size, size);
        // Allocate raw memory for carrier
        let ptr = mmap::map(carrier_layout)?;
        // Initialize carrier in memory
        let carrier = SlabCarrier::init(ptr.as_ptr(), size, size_class);
        // Return an unsafe ref to this carrier back to the caller
        Ok(carrier)
    }

    #[inline]
    unsafe fn alloc_sized(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        // Ensure allocated region has enough space for carrier header and aligned block
        let size = layout.size();
        if unlikely(size > Self::MAX_SIZE_CLASS.to_bytes()) {
            return Err(AllocError);
        }
        let size_class = self.size_class_for_unchecked(size);
        let index = self.index_for(size_class);
        let carriers = self.classes[index].read();
        for carrier in carriers.iter() {
            if let Ok(ptr) = carrier.alloc_block() {
                return Ok(NonNull::slice_from_raw_parts(ptr, size_class.to_bytes()));
            }
        }
        drop(carriers);
        // No carriers had availability, create a new carrier, locking
        // the carrier list for this size class while we do so, to avoid
        // other readers from trying to create their own carriers at the
        // same time
        let mut carriers = self.classes[index].write();
        let carrier_ptr = Self::create_carrier(size_class)?;
        let carrier = &mut *carrier_ptr;
        // This should never fail, but we only assert that in debug mode
        let result = carrier.alloc_block();
        debug_assert!(result.is_ok());
        carriers.push_front(UnsafeRef::from_raw(carrier_ptr));
        result.map(|data| NonNull::slice_from_raw_parts(data, size_class.to_bytes()))
    }

    #[inline]
    unsafe fn realloc_sized(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let new_size = new_layout.size();
        if unlikely(new_size > Self::MAX_SIZE_CLASS.to_bytes()) {
            return Err(AllocError);
        }
        let old_size = old_layout.size();
        let old_size_class = self.size_class_for_unchecked(old_size);
        let new_size = new_layout.size();
        let new_size_class = self.size_class_for_unchecked(new_size);
        // If the size is in the same size class, we don't have to do anything
        if old_size_class == new_size_class {
            return Ok(NonNull::slice_from_raw_parts(
                ptr,
                old_size_class.to_bytes(),
            ));
        }
        // Otherwise we have to allocate in the new size class,
        // copy to that new block, and deallocate the original block
        let non_null_byte_slice = self.allocate(new_layout)?;
        // Copy
        let copy_size = cmp::min(old_size, non_null_byte_slice.len());
        ptr::copy_nonoverlapping(ptr.as_ptr(), non_null_byte_slice.as_mut_ptr(), copy_size);
        // Deallocate the original block
        self.deallocate(ptr, old_layout);
        // Return new block
        Ok(non_null_byte_slice)
    }

    unsafe fn dealloc_sized(&self, ptr: NonNull<u8>, _layout: Layout) {
        // Locate the owning carrier and deallocate with it
        let raw = ptr.as_ptr();
        // Since the slabs are super-aligned, we can mask off the low
        // bits of the given pointer to find our carrier
        let carrier_ptr = superalign_down(raw as usize)
            as *mut SlabCarrier<LinkedListLink, ThreadSafeBlockBitSubset>;
        let carrier = &mut *carrier_ptr;
        carrier.free_block(raw);
    }
}

impl Drop for SegmentedAlloc {
    fn drop(&mut self) {
        // Drop single-block carriers
        let mut sbc = self.sbc.write();
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

        // Drop slab carriers
        let slab_size = SUPERALIGNED_CARRIER_SIZE;
        let slab_layout = unsafe { Layout::from_size_align_unchecked(slab_size, slab_size) };

        for class in self.classes.iter() {
            // Lock the class list while we free the slabs in it
            let mut list = class.write();
            // Collect the pointers/layouts for all slabs allocated in this class
            let mut slabs = list
                .iter()
                .map(|slab| (slab as *const _ as *mut _, slab_layout.clone()))
                .collect::<Vec<_>>();

            // Clear the list without dropping the elements, since we're handling that
            list.fast_clear();

            // Free the memory for all the slabs
            for (ptr, layout) in slabs.drain(..) {
                unsafe { mmap::unmap(ptr, layout) }
            }
        }
    }
}
unsafe impl Sync for SegmentedAlloc {}
unsafe impl Send for SegmentedAlloc {}
