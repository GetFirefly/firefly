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
    ) -> Result<*mut SlabCarrier<LinkedListLink, ThreadSafeBlockBitSubset>, AllocErr> {
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

unsafe impl AllocRef for SegmentedAlloc {
    #[inline]
    fn alloc(&mut self, layout: Layout, init: AllocInit) -> Result<MemoryBlock, AllocErr> {
        if layout.size() >= self.sbc_threshold {
            return unsafe { self.alloc_large(layout, init) };
        }
        unsafe { self.alloc_sized(layout, init) }
    }

    #[inline]
    unsafe fn grow(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
        placement: ReallocPlacement,
        init: AllocInit,
    ) -> Result<MemoryBlock, AllocErr> {
        if layout.size() >= self.sbc_threshold {
            // This was a single-block carrier
            //
            // Even if the new size would fit in a multi-block carrier, we're going
            // to keep the allocation in the single-block carrier, to avoid the extra
            // complexity, as there is little payoff
            return self.realloc_large(ptr, layout, new_size, placement, init);
        }

        self.realloc_sized(ptr, layout, new_size, placement, init)
    }

    #[inline]
    unsafe fn shrink(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
        placement: ReallocPlacement,
    ) -> Result<MemoryBlock, AllocErr> {
        if layout.size() >= self.sbc_threshold {
            // This was a single-block carrier
            //
            // Even if the new size would fit in a multi-block carrier, we're going
            // to keep the allocation in the single-block carrier, to avoid the extra
            // complexity, as there is little payoff
            return self.realloc_large(ptr, layout, new_size, placement, AllocInit::Uninitialized);
        }

        self.realloc_sized(ptr, layout, new_size, placement, AllocInit::Uninitialized)
    }

    #[inline]
    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() >= self.sbc_threshold {
            // This block would have to be in a single-block carrier
            return self.dealloc_large(ptr.as_ptr());
        }

        self.dealloc_sized(ptr, layout);
    }
}

impl SegmentedAlloc {
    /// This function handles allocations which exceed the single-block carrier threshold
    unsafe fn alloc_large(
        &mut self,
        layout: Layout,
        init: AllocInit,
    ) -> Result<MemoryBlock, AllocErr> {
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

        let block = MemoryBlock {
            ptr: NonNull::new_unchecked(data),
            size: data_layout_size,
        };
        AllocInit::init(init, block);

        // Return data pointer
        Ok(block)
    }

    unsafe fn realloc_large(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
        placement: ReallocPlacement,
        init: AllocInit,
    ) -> Result<MemoryBlock, AllocErr> {
        if placement != ReallocPlacement::MayMove {
            return Err(AllocErr);
        }

        // Allocate new carrier
        let block = self.alloc_large(
            Layout::from_size_align_unchecked(new_size, layout.align()),
            init,
        )?;
        // Copy old data into new carrier
        let old_ptr = ptr.as_ptr();
        let old_size = layout.size();
        ptr::copy_nonoverlapping(old_ptr, block.ptr.as_ptr(), cmp::min(old_size, block.size));
        // Free old carrier
        self.dealloc_large(old_ptr);
        // Return new carrier
        Ok(block)
    }

    /// This function handles allocations which exceed the single-block carrier threshold
    unsafe fn dealloc_large(&mut self, ptr: *const u8) {
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
    ) -> Result<*mut SlabCarrier<LinkedListLink, ThreadSafeBlockBitSubset>, AllocErr> {
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
    unsafe fn alloc_sized(
        &mut self,
        layout: Layout,
        init: AllocInit,
    ) -> Result<MemoryBlock, AllocErr> {
        // Ensure allocated region has enough space for carrier header and aligned block
        let size = layout.size();
        if unlikely(size > Self::MAX_SIZE_CLASS.to_bytes()) {
            return Err(AllocErr);
        }
        let size_class = self.size_class_for_unchecked(size);
        let index = self.index_for(size_class);
        let carriers = self.classes[index].read();
        for carrier in carriers.iter() {
            if let Ok(ptr) = carrier.alloc_block() {
                let block = MemoryBlock {
                    ptr,
                    size: size_class.to_bytes(),
                };
                AllocInit::init(init, block);
                return Ok(block);
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
        let block = result.map(|ptr| MemoryBlock {
            ptr,
            size: size_class.to_bytes(),
        })?;
        AllocInit::init(init, block);
        Ok(block)
    }

    #[inline]
    unsafe fn realloc_sized(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
        placement: ReallocPlacement,
        init: AllocInit,
    ) -> Result<MemoryBlock, AllocErr> {
        if unlikely(new_size > Self::MAX_SIZE_CLASS.to_bytes()) {
            return Err(AllocErr);
        }
        let size = layout.size();
        let size_class = self.size_class_for_unchecked(size);
        let new_size_class = self.size_class_for_unchecked(new_size);
        // If the size is in the same size class, we don't have to do anything
        if size_class == new_size_class {
            return Ok(MemoryBlock {
                ptr,
                size: size_class.to_bytes(),
            });
        }
        // Otherwise we have to allocate in the new size class,
        // copy to that new block, and deallocate the original block
        if placement != ReallocPlacement::MayMove {
            return Err(AllocErr);
        }
        let align = layout.align();
        let new_layout = Layout::from_size_align_unchecked(new_size, align);
        let block = self.alloc(new_layout, init)?;
        // Copy
        let copy_size = cmp::min(size, block.size);
        ptr::copy_nonoverlapping(ptr.as_ptr(), block.ptr.as_ptr(), copy_size);
        // Deallocate the original block
        self.dealloc(ptr, layout);
        // Return new block
        AllocInit::init_offset(init, block, copy_size);
        Ok(block)
    }

    unsafe fn dealloc_sized(&mut self, ptr: NonNull<u8>, _layout: Layout) {
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
