//! TODO: WIP
use core::ptr::NonNull;
use core::alloc::{Alloc, AllocErr, Layout};

use crate::std_alloc::SingleBlockCarrierList;
use crate::size_classes::{MAX_SIZE_CLASS, SizeClasses};

/// Like `StandardAlloc`, `FixedAlloc` splits allocations into two major categories,
/// multi-block carriers up to a certain threshold, after which allocations use single-block carriers.
///
/// However, `FixedAlloc` differs in some key ways:
///
/// - The multi-block carriers are allocated into multiple size classes, where each carrier
///   belongs to a size class and therefore only fulfills allocation requests for blocks of
///   uniform size.
/// - The single-block carrier threshold is statically determined based on the maximum size
///   class for the multi-block carriers and is therefore not configurable.
///
/// Each size class for multi-block carriers contains at least one carrier, and new carriers are
/// allocated as needed when the carrier(s) for a size class are unable to fulfill allocation requests.
///
/// Allocations of blocks in multi-block carriers are filled using address order for both carriers
/// and blocks to reduce fragmentation and improve allocation locality for allocations that fall
/// within the same size class.
pub struct FixedAlloc {
    sbc: SingleBlockCarrierList,
    mbc: SizeClasses,
}
impl FixedAlloc {
    #[inline]
    pub fn new() -> Self {
        Self {
            sbc: SingleBlockCarrierList::new(),
            mbc: SizeClasses::new(),
        }
    }
}

unsafe impl Alloc for FixedAlloc {
    unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocErr> {
        unimplemented!()
    }

    unsafe fn realloc(&mut self, ptr: NonNull<u8>, layout: Layout, new_size: usize) -> Result<NonNull<u8>, AllocErr> {
        unimplemented!()
    }

    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        unimplemented!();
    }
}
