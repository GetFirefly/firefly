use alloc::fmt::{self, Debug, Formatter};

use super::{FreeBlock, FreeBlockRef};

/// This struct is similar in nature to `Block`, but is used
/// to store the size of the preceding data region when the block is free.
///
/// When the next neighboring block is freed, a check is performed to
/// see if it can be combined with its preceding block, the one containing
/// this footer, this combining operation is called "coalescing".
///
/// If the blocks can be coalesced, then the footer is used to get the size of this
/// block, so that the address of the block header can be calculated. That address
/// is then used to access the header and update it with new metadata
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct BlockFooter(usize);
impl BlockFooter {
    #[inline]
    pub const fn new(size: usize) -> Self {
        Self(size)
    }

    #[allow(unused)]
    #[inline]
    pub fn usable_size(&self) -> usize {
        self.0
    }

    #[allow(unused)]
    #[inline]
    pub unsafe fn to_block(&self) -> FreeBlockRef {
        let raw = self as *const _ as *const u8;
        let header_offset = (self.0 as isize) * -1;
        FreeBlockRef::from_raw(raw.offset(header_offset) as *mut FreeBlock)
    }
}

impl Debug for BlockFooter {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("BlockFooter")
            .field("size", &self.0)
            .finish()
    }
}
