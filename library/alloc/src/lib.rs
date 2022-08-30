#![no_std]
// Used for allocators
#![feature(allocator_api)]
#![feature(nonnull_slice_from_raw_parts)]
#![feature(slice_ptr_get)]
#![feature(slice_ptr_len)]
// Used for access to pointer metadata
#![feature(ptr_metadata)]
#![feature(pointer_byte_offsets)]
// Used for access to size/alignment information for raw pointers
#![feature(layout_for_ptr)]
// Used for the Unsize marker trait
#![feature(unsize)]
// Used for let PATTERN = EXPR else { .. }
#![feature(let_else)]
// Used for implementing From<&[T]> for GcBox
#![feature(maybe_uninit_write_slice)]
// Used for implementing Fn, etc. for GcBox, as
// well as interop with closures produced by the compiler
#![feature(fn_traits)]
#![feature(unboxed_closures)]
// For specializing the WriteCloneIntoRaw trait
#![feature(min_specialization)]
// Used for const TypeId::of::<T>()
#![feature(const_type_id)]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;
#[cfg(test)]
extern crate test;

pub mod allocators;
pub mod fragment;
pub mod gc;
pub mod heap;
pub mod mmap;
pub mod rc;
mod utils;

pub use self::utils::*;

/// Based on the trait of the same name in the standard library alloc crate,
/// specializes clones into pre-allocated, uninitialized memory.
///
/// Used by `RcBox::make_mut` and `GcBox::clone`
pub(crate) trait WriteCloneIntoRaw: Sized {
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
