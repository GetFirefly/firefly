use core::alloc::Layout;

use intrusive_collections::RBTree;
use intrusive_collections::rbtree;

use crate::sorted::{SortedKeyAdapter, SortOrder};

use super::{FreeBlock, FreeBlockRef};

/// Used within FreeBlockTree
pub type FreeBlockTree = RBTree<SortedKeyAdapter<FreeBlock>>;

/// This struct maintains an ordered tree of free blocks.
///
/// The structure internally contains two red-black trees, one sorted by address order,
/// and the other by a user-selected sort order, this sorting determines the
/// order in which free blocks are selected for use by a carrier during allocation.
pub struct FreeBlocks {
    // address-ordered tree
    addr_tree: FreeBlockTree,
    // custom-ordered tree
    user_tree: FreeBlockTree,
    // the ordering used by `user_tree`
    order: SortOrder,
}
impl FreeBlocks {
    #[inline]
    pub fn new(order: SortOrder) -> Self {
        Self {
            addr_tree: RBTree::new(SortedKeyAdapter::new(SortOrder::AddressOrder)),
            user_tree: RBTree::new(SortedKeyAdapter::new(order)),
            order,
        }
    }

    /// Get the number of blocks in this tree
    #[cfg(test)]
    #[inline]
    pub fn count(&self) -> usize {
        self.iter().count()
    }

    #[cfg(test)]
    #[inline]
    pub fn iter(&self) -> rbtree::Iter<'_, SortedKeyAdapter<FreeBlock>> {
        self.addr_tree.iter()
    }

    /// Lookup a free block, choosing the first one that fits the given request, preferring
    /// lower-addressed blocks before higher-addressed blocks.
    #[allow(unused)]
    pub fn find_first_fit(&self, layout: &Layout) -> Option<FreeBlockRef> {
        let mut cursor = self.addr_tree.front();
        let mut result = None;

        let aligned = layout.pad_to_align().unwrap();
        let requested = aligned.size();

        while let Some(block) = cursor.get() {
            let usable = block.usable_size();

            if usable >= requested {
                result = Some(unsafe { FreeBlockRef::from_raw(block as *const _ as *mut _) });
                break;
            }

            cursor.move_next();
        }

        result
    }

    /// Lookup a free block which is the best fit for the given requested size
    pub fn find_best_fit(&self, layout: &Layout) -> Option<FreeBlockRef> {
        let mut cursor = self.user_tree.front();
        let mut result = None;
        let mut best_size = 0;

        let aligned = layout.pad_to_align().unwrap();
        let requested = aligned.size();

        match self.order {
            SortOrder::AddressOrder => {
                while let Some(block) = cursor.get() {
                    // When in AddressOrder, we have to search the whole tree for the best fit
                    let usable = block.usable_size();

                    // Not suitable
                    if usable < requested {
                        cursor.move_next();
                        continue;
                    }

                    // If we've found a better fit, or don't yet have a fit
                    // mark the current node as the current best fit
                    if usable < best_size || result.is_none() {
                        result = Some(unsafe { FreeBlockRef::from_raw(block as *const _ as *mut _) });
                        best_size = usable;
                    }

                    cursor.move_next();
                }
            }
            SortOrder::SizeAddressOrder => {
                while let Some(block) = cursor.get() {
                    // A best fit can be found as the previous neighbor of the first block which is
                    // too small or the last block in the tree, if all blocks
                    // are of adequate size
                    let usable = block.usable_size();
                    if usable < requested {
                        break;
                    }

                    result = Some(unsafe { FreeBlockRef::from_raw(block as *const _ as *mut _) });
                    cursor.move_next();
                }
            }
        }

        result
    }

    /// Inserts the given block into this tree
    pub unsafe fn insert(&mut self, block: FreeBlockRef) {
        let _ = self.addr_tree.insert(block.into());
        let _ = self.user_tree.insert(block.into());
    }

    /// Removes the given block from this tree
    pub unsafe fn remove(&mut self, block: FreeBlockRef) {
        // remove from address-ordered tree
        assert!(block.addr_link.is_linked());
        if block.addr_link.is_linked() {
            let mut cursor = self.addr_tree.cursor_mut_from_ptr(block.as_ptr());
            let removed = cursor.remove();
            assert!(removed.is_some());
        }

        assert!(block.user_link.is_linked());
        // remove from user-ordered tree
        if block.user_link.is_linked() {
            let mut cursor = self.user_tree.cursor_mut_from_ptr(block.as_ptr());
            let removed = cursor.remove();
            assert!(removed.is_some());
        }
    }
}

