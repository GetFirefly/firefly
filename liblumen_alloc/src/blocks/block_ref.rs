use core::borrow::{Borrow, BorrowMut};
use core::convert::TryFrom;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

use intrusive_collections::UnsafeRef;

use super::{Block, FreeBlock};

/// Unchecked shared pointer type
///
/// This type is essentially equivalent to a reference-counted
/// smart pointer, e.g. `Rc`/`Arc`, except that no reference count
/// is maintained. Instead the user of `BlockRef` must make sure
/// that the object pointed to is freed manually when no longer needed.
///
/// Guarantees that are expected to be upheld by users of `BlockRef`:
///
/// - The object pointed to by `BlockRef` is not moved or dropped
/// while a `BlockRef` points to it. This will result in use-after-free,
/// or undefined behavior.
/// - Additionally, you must not allow a mutable reference to be held
/// or created while a `BlockRef` pointing to the same object is in use,
/// unless special care is taken to ensure that only one or the other is
/// used to access the underlying object at any given time.
///
/// For example, if you create a `Block`, then create a `BlockRef` pointing
/// to that block, then subsequently obtain a mutable reference to the block
/// via the `BlockRef` or using standard means, then it must never be the
/// case that there are other active references pointing to the same object while
/// the mutable reference is in use.
///
/// NOTE: This type is used internally to make block management more ergonomic
/// while avoiding the use of raw pointers. It can be used in lieu of either
/// raw pointers or references, as it derefs to the `Block` it points to. Since
/// we have an extra flag bit in `Block` available, we could use it to track whether
/// more than one mutable reference has been created to the same block, and panic,
/// but currently that is not the case.
#[derive(Debug, Copy, PartialEq)]
#[repr(transparent)]
pub struct BlockRef(NonNull<Block>);

// Conversions

impl From<FreeBlockRef> for BlockRef {
    #[inline]
    fn from(block_ref: FreeBlockRef) -> Self {
        Self(block_ref.0.cast())
    }
}
impl From<&Block> for BlockRef {
    #[inline]
    fn from(block: &Block) -> Self {
        let nn = unsafe { NonNull::new_unchecked(block as *const _ as *mut _) };
        Self(nn)
    }
}
impl From<&mut Block> for BlockRef {
    #[inline]
    fn from(block: &mut Block) -> Self {
        let nn = unsafe { NonNull::new_unchecked(block as *mut _) };
        Self(nn)
    }
}
impl From<UnsafeRef<Block>> for BlockRef {
    #[inline]
    fn from(block: UnsafeRef<Block>) -> Self {
        let raw = UnsafeRef::into_raw(block) as *mut Block;
        let nn = unsafe { NonNull::new_unchecked(raw) };
        Self(nn)
    }
}
impl From<UnsafeRef<FreeBlock>> for BlockRef {
    #[inline]
    fn from(block: UnsafeRef<FreeBlock>) -> Self {
        let raw = UnsafeRef::into_raw(block) as *const _ as *mut Block;
        let nn = unsafe { NonNull::new_unchecked(raw) };
        Self(nn)
    }
}

impl BlockRef {
    /// Creates a new `BlockRef` from a raw pointer.
    ///
    /// NOTE: This is very unsafe! You must make sure that
    /// the object being pointed to will not live longer than
    /// the `BlockRef` returned, or use-after-free will result.
    #[inline]
    pub unsafe fn from_raw(ptr: *const Block) -> Self {
        Self(NonNull::new_unchecked(ptr as *mut _))
    }

    /// Converts a `BlockRef` into the underlying raw pointer.
    #[allow(unused)]
    #[inline]
    pub fn into_raw(ptr: Self) -> *mut Block {
        ptr.0.as_ptr()
    }

    /// Gets the underlying pointer from the `BlockRef`
    #[allow(unused)]
    #[inline]
    pub fn as_ptr(&self) -> *mut Block {
        self.0.as_ptr()
    }
}
impl Clone for BlockRef {
    #[inline]
    fn clone(&self) -> BlockRef {
        Self(self.0)
    }
}
impl Deref for BlockRef {
    type Target = Block;

    #[inline]
    fn deref(&self) -> &Block {
        self.as_ref()
    }
}
impl DerefMut for BlockRef {
    #[inline]
    fn deref_mut(&mut self) -> &mut Block {
        self.as_mut()
    }
}
impl AsRef<Block> for BlockRef {
    #[inline]
    fn as_ref(&self) -> &Block {
        unsafe { self.0.as_ref() }
    }
}
impl AsMut<Block> for BlockRef {
    #[inline]
    fn as_mut(&mut self) -> &mut Block {
        unsafe { self.0.as_mut() }
    }
}
impl Borrow<Block> for BlockRef {
    #[inline]
    fn borrow(&self) -> &Block {
        self.as_ref()
    }
}
impl BorrowMut<Block> for BlockRef {
    #[inline]
    fn borrow_mut(&mut self) -> &mut Block {
        self.as_mut()
    }
}
unsafe impl Send for BlockRef {}
unsafe impl Sync for BlockRef {}

/// Same as `BlockRef`, but for `FreeBlock`
///
/// In many cases, it is fine to use `BlockRef` to
/// represent references to `FreeBlock`, but for safety
/// we want to ensure that we don't try to treat a
/// `Block` as a `FreeBlock` when that block was never
/// initialized as a free block, or has since been overwritten
/// with user data.
///
/// By using `FreeBlockRef` in conjunction with `BlockRef`,
/// providing safe conversions between them, as well as unsafe
/// conversions for the few occasions where that is necessary,
/// we can provide some compile-time guarantees that protect us
/// against simple mistakes
#[derive(Debug, Copy, PartialEq)]
#[repr(transparent)]
pub struct FreeBlockRef(NonNull<FreeBlock>);

// Conversions

impl From<&FreeBlock> for FreeBlockRef {
    #[inline]
    fn from(block: &FreeBlock) -> FreeBlockRef {
        let nn = unsafe { NonNull::new_unchecked(block as *const _ as *mut _) };
        FreeBlockRef(nn)
    }
}
impl From<&mut FreeBlock> for FreeBlockRef {
    #[inline]
    fn from(block: &mut FreeBlock) -> FreeBlockRef {
        let nn = unsafe { NonNull::new_unchecked(block as *mut _) };
        FreeBlockRef(nn)
    }
}
impl From<UnsafeRef<FreeBlock>> for FreeBlockRef {
    #[inline]
    fn from(block: UnsafeRef<FreeBlock>) -> FreeBlockRef {
        let raw = UnsafeRef::into_raw(block);
        let nn = unsafe { NonNull::new_unchecked(raw) };
        FreeBlockRef(nn)
    }
}
impl Into<UnsafeRef<FreeBlock>> for FreeBlockRef {
    #[inline]
    fn into(self) -> UnsafeRef<FreeBlock> {
        let raw = FreeBlockRef::into_raw(self);
        unsafe { UnsafeRef::from_raw(raw) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TryFromBlockError;

impl TryFrom<&Block> for FreeBlockRef {
    type Error = TryFromBlockError;

    #[inline]
    fn try_from(block: &Block) -> Result<Self, Self::Error> {
        if !block.is_free() {
            return Err(TryFromBlockError);
        }
        Ok(unsafe { FreeBlockRef::from_raw(block as *const _ as *mut FreeBlock) })
    }
}
impl TryFrom<&mut Block> for FreeBlockRef {
    type Error = TryFromBlockError;

    #[inline]
    fn try_from(block: &mut Block) -> Result<Self, Self::Error> {
        if !block.is_free() {
            return Err(TryFromBlockError);
        }
        Ok(unsafe { FreeBlockRef::from_raw(block as *const _ as *mut FreeBlock) })
    }
}
impl TryFrom<BlockRef> for FreeBlockRef {
    type Error = TryFromBlockError;

    #[inline]
    fn try_from(block: BlockRef) -> Result<Self, Self::Error> {
        if !block.is_free() {
            return Err(TryFromBlockError);
        }
        Ok(FreeBlockRef(block.0.cast()))
    }
}

impl FreeBlockRef {
    /// Creates a new `FreeBlockRef` from a raw pointer.
    ///
    /// NOTE: This is very unsafe! You must make sure that
    /// the object being pointed to will not live longer than
    /// the `FreeBlockRef` returned, or use-after-free will result.
    #[inline]
    pub unsafe fn from_raw(ptr: *const FreeBlock) -> Self {
        Self(NonNull::new_unchecked(ptr as *mut _))
    }

    /// Converts a `BlockRef` into the underlying raw pointer.
    #[inline]
    pub fn into_raw(ptr: Self) -> *mut FreeBlock {
        ptr.0.as_ptr()
    }

    /// Gets the underlying pointer from the `FreeBlockRef`
    #[inline]
    pub fn as_ptr(&self) -> *mut FreeBlock {
        self.0.as_ptr()
    }
}
impl Clone for FreeBlockRef {
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0)
    }
}
impl Deref for FreeBlockRef {
    type Target = FreeBlock;

    #[inline]
    fn deref(&self) -> &FreeBlock {
        self.as_ref()
    }
}
impl DerefMut for FreeBlockRef {
    #[inline]
    fn deref_mut(&mut self) -> &mut FreeBlock {
        self.as_mut()
    }
}
impl AsRef<Block> for FreeBlockRef {
    #[inline]
    fn as_ref(&self) -> &Block {
        let raw = self.0.as_ptr() as *mut Block;
        unsafe { &*raw }
    }
}
impl AsRef<FreeBlock> for FreeBlockRef {
    #[inline]
    fn as_ref(&self) -> &FreeBlock {
        unsafe { self.0.as_ref() }
    }
}
impl AsMut<Block> for FreeBlockRef {
    #[inline]
    fn as_mut(&mut self) -> &mut Block {
        let raw = self.0.as_ptr() as *mut Block;
        unsafe { &mut *raw }
    }
}
impl AsMut<FreeBlock> for FreeBlockRef {
    #[inline]
    fn as_mut(&mut self) -> &mut FreeBlock {
        unsafe { self.0.as_mut() }
    }
}
impl Borrow<Block> for FreeBlockRef {
    #[inline]
    fn borrow(&self) -> &Block {
        self.as_ref()
    }
}
impl Borrow<FreeBlock> for FreeBlockRef {
    #[inline]
    fn borrow(&self) -> &FreeBlock {
        self.as_ref()
    }
}
impl BorrowMut<Block> for FreeBlockRef {
    #[inline]
    fn borrow_mut(&mut self) -> &mut Block {
        self.as_mut()
    }
}
impl BorrowMut<FreeBlock> for FreeBlockRef {
    #[inline]
    fn borrow_mut(&mut self) -> &mut FreeBlock {
        self.as_mut()
    }
}
unsafe impl Send for FreeBlockRef {}
unsafe impl Sync for FreeBlockRef {}
