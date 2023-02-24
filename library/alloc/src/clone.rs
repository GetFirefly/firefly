/// Based on the trait of the same name in the standard library alloc crate,
/// specializes clones into pre-allocated, uninitialized memory.
///
/// Used by `RcBox::make_mut` and `GcBox::clone`
pub trait WriteCloneIntoRaw: Sized {
    unsafe fn write_clone_into_raw(&self, target: *mut Self);
}

impl<T: Clone> WriteCloneIntoRaw for T {
    #[inline]
    default unsafe fn write_clone_into_raw(&self, target: *mut Self) {
        target.write(self.clone());
    }
}

impl<T: Copy> WriteCloneIntoRaw for T {
    #[inline]
    unsafe fn write_clone_into_raw(&self, target: *mut Self) {
        target.copy_from_nonoverlapping(self, 1);
    }
}
