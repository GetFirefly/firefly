mod multi_block;
mod single_block;
mod slab;

pub use multi_block::MultiBlockCarrier;
pub use single_block::SingleBlockCarrier;
pub use slab::SlabCarrier;

use cfg_if::cfg_if;

cfg_if! {
    // We use a smaller super-alignment on wasm32, one that happens to match
    // the page size, which is convenient. The reasoning is that in a few allocators,
    // we allocate a set of initial buckets, each with a single super-aligned carrier,
    // and these add up. By keeping the super-aligned size smaller, we apply less
    // memory pressure
    if #[cfg(target_arch = "wasm32")] {
        /// The number of bits to shift/mask to find a superaligned address
        pub const SUPERALIGNED_BITS: usize = 16;
        /// The number of bits to shift to find a superaligned carrier address
        pub const SUPERALIGNED_CARRIER_SHIFT: usize = SUPERALIGNED_BITS;
        /// The size of a superaligned carrier, 64k (65,536 bytes)
        pub const SUPERALIGNED_CARRIER_SIZE: usize = 1usize << SUPERALIGNED_CARRIER_SHIFT;
        /// The mask needed to go from a pointer in a SA carrier to the carrier
        pub const SUPERALIGNED_CARRIER_MASK: usize = (!0usize) << SUPERALIGNED_CARRIER_SHIFT;
    } else {
        /// The number of bits to shift/mask to find a superaligned address
        pub const SUPERALIGNED_BITS: usize = 18;
        /// The number of bits to shift to find a superaligned carrier address
        pub const SUPERALIGNED_CARRIER_SHIFT: usize = SUPERALIGNED_BITS;
        /// The size of a superaligned carrier, 262k (262,144 bytes)
        pub const SUPERALIGNED_CARRIER_SIZE: usize = 1usize << SUPERALIGNED_CARRIER_SHIFT;
        /// The mask needed to go from a pointer in a SA carrier to the carrier
        pub const SUPERALIGNED_CARRIER_MASK: usize = (!0usize) << SUPERALIGNED_CARRIER_SHIFT;
    }
}

/// Get the nearest super-aligned address by rounding down
#[inline(always)]
pub fn superalign_down(addr: usize) -> usize {
    addr & SUPERALIGNED_CARRIER_MASK
}

/// Get the nearest super-aligned address by rounding up
#[allow(unused)]
#[inline(always)]
pub fn superalign_up(addr: usize) -> usize {
    superalign_down(addr + !SUPERALIGNED_CARRIER_MASK)
}

use intrusive_collections::{intrusive_adapter, UnsafeRef};
use intrusive_collections::{LinkedList, LinkedListLink};
use intrusive_collections::{RBTree, RBTreeLink};

use crate::blocks::ThreadSafeBlockBitSubset;
use crate::sorted::SortedKeyAdapter;

// Type alias for the list of currently allocated single-block carriers
pub(crate) type SingleBlockCarrierList = LinkedList<SingleBlockCarrierListAdapter>;
// Type alias for the ordered tree of currently allocated multi-block carriers
pub(crate) type MultiBlockCarrierTree = RBTree<SortedKeyAdapter<MultiBlockCarrier<RBTreeLink>>>;
// Type alias for the list of currently allocated slab carriers
pub(crate) type SlabCarrierList = LinkedList<SlabCarrierListAdapter<ThreadSafeBlockBitSubset>>;

// Implementation of adapter for intrusive collection used for slab carriers
intrusive_adapter!(pub SlabCarrierListAdapter<S> = UnsafeRef<SlabCarrier<LinkedListLink, S>>: SlabCarrier<LinkedListLink, S> { link: LinkedListLink });

// Implementation of adapter for intrusive collection used for single-block carriers
intrusive_adapter!(pub SingleBlockCarrierListAdapter = UnsafeRef<SingleBlockCarrier<LinkedListLink>>: SingleBlockCarrier<LinkedListLink> { link: LinkedListLink });
