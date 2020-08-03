mod block;
mod block_bit_set;
mod block_footer;
mod block_ref;
mod free_block;
mod free_block_tree;

pub use self::block::Block;
pub use self::block_bit_set::*;
pub use self::block_footer::BlockFooter;
pub use self::block_ref::{BlockRef, FreeBlockRef};
pub use self::free_block::FreeBlock;
pub use self::free_block_tree::{FreeBlockTree, FreeBlocks};

#[cfg(test)]
mod tests {
    use core::alloc::AllocInit;
    use core::mem;
    use core::ptr;

    use liblumen_core::alloc::{prelude::*, SysAlloc};
    use liblumen_core::sys::sysconf;

    use crate::sorted::SortOrder;

    use super::*;

    #[test]
    fn block_meta_api_test() {
        let size = sysconf::pagesize();
        let usable = size - mem::size_of::<Block>();
        let layout = Layout::from_size_align(size, size).unwrap();
        let alloc_block = SysAlloc::get_mut()
            .alloc(layout.clone(), AllocInit::Uninitialized)
            .expect("unable to map memory");
        let raw = alloc_block.ptr.as_ptr() as *mut Block;
        unsafe {
            ptr::write(raw, Block::new(usable));
        }
        let block = unsafe { &mut *raw };
        // Can update size
        block.set_size(size - 8);
        assert_eq!(block.usable_size(), size - 8);
        block.set_size(usable);
        assert_eq!(block.usable_size(), usable);
        // Can toggle allocation status
        block.set_allocated();
        assert!(!block.is_free());
        block.set_free();
        assert!(block.is_free());
        // Can toggle last status
        block.set_last();
        assert!(block.is_last());
        block.clear_last();
        assert!(!block.is_last());
        block.set_last();
        // Can toggle prev free status
        assert!(!block.is_prev_free());
        block.set_prev_free();
        assert!(block.is_prev_free());
        block.set_prev_allocated();
        assert!(!block.is_prev_free());
        // Make sure we don't read uninitialized memory
        assert_eq!(block.next(), None);

        // Cleanup
        unsafe { SysAlloc::get_mut().dealloc(alloc_block.ptr, layout) };
    }

    #[test]
    fn block_try_alloc_test() {
        let size = sysconf::pagesize();
        let usable = size - mem::size_of::<Block>();
        let layout = Layout::from_size_align(size, size).unwrap();
        let alloc_block = SysAlloc::get_mut()
            .alloc(layout.clone(), AllocInit::Uninitialized)
            .expect("unable to map memory");
        let raw = alloc_block.ptr.as_ptr() as *mut FreeBlock;
        // Block is free, and last
        let mut header = Block::new(usable);
        header.set_free();
        header.set_last();
        unsafe {
            ptr::write(raw, FreeBlock::from_block(header));
        }
        let free = unsafe { &mut *raw };
        assert_eq!(free.usable_size(), usable);
        // Try allocate entire usable size
        let request_layout = Layout::from_size_align(usable, mem::size_of::<usize>()).unwrap();
        let result = free.try_alloc(&request_layout);
        assert!(result.is_ok());
        // Block should no longer be free
        assert!(!free.is_free());
        // Another attempt to allocate this block should fail
        let result = free.try_alloc(&request_layout);
        assert!(result.is_err());
        let mut allocated = unsafe { BlockRef::from_raw(free as *mut _ as *mut Block) };
        assert_eq!(allocated.usable_size(), usable);
        // Free the block
        allocated.free();
        assert!(allocated.is_free());
        // Should have a block footer now
        let result = allocated.footer();
        assert!(result.is_some());
        let result = result.unwrap();
        let footer = unsafe { result.as_ref() };
        assert_eq!(footer.usable_size(), usable);
        // Another allocation in this block will succeed
        let result = free.try_alloc(&request_layout);
        assert!(result.is_ok());

        // Cleanup
        unsafe { SysAlloc::get_mut().dealloc(alloc_block.ptr, layout) };
    }

    #[test]
    fn block_free_block_tree_test() {
        // Allocate space for two blocks, each page sized, but we're going to treat
        // the latter block as half that size
        let size = sysconf::pagesize() * 2;
        let usable = sysconf::pagesize() - mem::size_of::<Block>();
        let layout = Layout::from_size_align(size, size).unwrap();
        let alloc_block = SysAlloc::get_mut()
            .alloc(layout.clone(), AllocInit::Uninitialized)
            .expect("unable to map memory");
        // Get pointers to both blocks
        let raw = alloc_block.ptr.as_ptr() as *mut FreeBlock;
        let raw2 = unsafe { (raw as *mut u8).add(sysconf::pagesize()) as *mut FreeBlock };
        // Write block headers
        unsafe {
            let mut block1 = Block::new(usable);
            block1.set_free();
            let mut block2 = Block::new(usable / 2);
            block2.set_free();
            block2.set_prev_free();
            block2.set_last();
            ptr::write(raw, FreeBlock::from_block(block1));
            ptr::write(raw2, FreeBlock::from_block(block2));
        }
        // Get blocks
        let fblock1 = unsafe { FreeBlockRef::from_raw(raw) };
        let fblock2 = unsafe { FreeBlockRef::from_raw(raw2) };
        // Need a sub-region here so we can clean up the mapped memory at the end
        // without a segmentation fault due to dropping the tree after the memory
        // is unmapped
        {
            // Create empty tree
            let mut tree = FreeBlocks::new(SortOrder::SizeAddressOrder);
            // Add blocks to tree
            unsafe {
                tree.insert(fblock1);
                tree.insert(fblock2);
            }
            // Find free block, we should get fblock2, since best fit would be the
            // smallest block which fits the request, and our tree should be sorted by
            // size
            let req_size = 1024;
            let request_layout =
                Layout::from_size_align(req_size, mem::size_of::<usize>()).unwrap();
            let result = tree.find_best_fit(&request_layout);
            assert!(result.is_some());
            let result_block = result.unwrap();
            assert_eq!(
                result_block.as_ptr() as *const u8,
                fblock2.as_ptr() as *const u8
            );
        }
        unsafe { SysAlloc::get_mut().dealloc(alloc_block.ptr, layout) };
    }
}
