use liblumen_core::alloc::Layout;

use crate::sorted::Link;

/// This struct is the carrier type for large allocations that
/// exceed a given threshold, typically anything larger than
/// `size_classes::MAX_SIZE_CLASS`.
///
/// This type of carrier only contains a single block, and is
/// optimized for that case.
///
/// NOTE: Single-block carriers are currently freed when the
/// block they contain is freed, but it may be that we will want
/// to cache some number of these carriers if large allocations
/// are frequent and reuse known to be likely
#[repr(C)]
pub struct SingleBlockCarrier<L: Link> {
    pub(crate) size: usize,
    pub(crate) layout: Layout,
    pub(crate) link: L,
}
impl<L> SingleBlockCarrier<L>
where
    L: Link,
{
    /// Returns the Layout used for the data contained in this carrier
    #[inline]
    pub fn layout(&self) -> Layout {
        self.layout.clone()
    }

    /// Returns a raw pointer to the data region in the sole block of this carrier
    ///
    /// NOTE: You may get the same pointer by asking a Block for its data, but
    /// this function bypasses the need to get the block and then ask for the data.
    /// This is "safe" since the data location is known by the carrier, or at least
    /// is able to be calculated by the carrier.
    ///
    /// NOTE: This is unsafe because 1.) the pointer is unmanaged, and 2.) if the
    /// pointer or a reference constructed from that pointer lives longer than this
    /// carrier, then it will be invalid, potentially allowing use-after-free.
    /// Additionally, the data pointer is the pointer given to `free`, so there is
    /// the potential that multiple pointers could permit double-free scenarios.
    /// This is less of a risk as the current implementation only frees memory if
    /// an allocated carrier can be found which owns the pointer, and after the first
    /// free, that can't happen..
    #[allow(unused)]
    #[inline]
    pub unsafe fn data<T>(&self) -> *const T {
        let (_layout, data_offset) = Layout::new::<Self>().extend(self.layout.clone()).unwrap();

        let ptr = self as *const _ as *const u8;
        ptr.add(data_offset) as *const T
    }

    /// Calculate the usable size of this carrier, specifically the size
    /// of the data region contained in this carrier's block
    #[allow(unused)]
    #[inline]
    pub fn usable_size(&self) -> usize {
        self.size - (self.size - self.layout.size())
    }

    /// Determines if the given pointer belongs to this carrier, this is
    /// primarily used in `free` to determine which carrier to free.
    #[inline]
    pub fn owns(&self, ptr: *const u8) -> bool {
        let this = self as *const _ as usize;
        let ptr = ptr as usize;

        // Belongs to a lower-addressed carrier
        if ptr < this {
            return false;
        }

        // Belongs to a higher-addressed carrier
        if (ptr - this) > self.size {
            return false;
        }

        // Falls within this carrier's address range
        true
    }
}
