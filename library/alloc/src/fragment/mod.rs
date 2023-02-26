use alloc::alloc::{AllocError, Allocator, Global, Layout};
use alloc::boxed::Box;
use core::cell::UnsafeCell;
use core::cmp;
use core::fmt;
use core::ops::Range;
use core::ptr::{self, NonNull};

use intrusive_collections::intrusive_adapter;
use intrusive_collections::{LinkedList, LinkedListLink, UnsafeRef};

use firefly_system::MIN_ALIGN;

use crate::heap::Heap;

intrusive_adapter!(pub HeapFragmentAdapter = UnsafeRef<HeapFragment>: HeapFragment { link: LinkedListLink });

/// A type alias for the intrusive linked list type for storing heap fragments
pub type HeapFragmentList = LinkedList<HeapFragmentAdapter>;

/// A low-level fragment type which is represented by a pointer to a region of
/// allocated heap memory, and the layout of the type stored in that region.
///
/// This is not really intended to be used directly, you should prefer [`HeapFragment`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawFragment {
    layout: Layout,
    base: NonNull<u8>,
}
impl RawFragment {
    /// Get a pointer to the data in this heap fragment
    #[inline]
    pub fn data(&self) -> NonNull<u8> {
        self.base
    }

    /// Return a pointer range representing the addressable memory of this fragment
    #[inline]
    pub fn as_ptr_range(&self) -> Range<*mut u8> {
        let base = self.base.as_ptr();
        let end = unsafe { base.add(self.layout.size()) };
        base..end
    }

    /// Get the layout of this heap fragment
    #[inline]
    pub fn layout(&self) -> Layout {
        self.layout.clone()
    }
}

/// Represents a chunk of allocated heap memory which can be used as an [`Allocator`].
///
/// A heap fragment may also have a destructor associated with it, which is used to ensure
/// that types with a [`Drop`] impl are properly executed when dropping the fragment. In
/// cases where a destructor is used, it is not safe to allocate into the fragment after
/// the destructor is set, even if the allocated type has no [`Drop`] impl. To defend against
/// such improper use, the allocator implementation will panic if an attempt is made to allocate
/// into a heap with a destructor.
pub struct HeapFragment {
    /// Link to the intrusive list that holds all heap fragments
    pub link: LinkedListLink,
    /// The memory region allocated for this fragment
    raw: RawFragment,
    /// A pointer to the top of the allocated region of this fragment,
    /// e.g. when the fragment is unused, `top == raw.base`
    top: UnsafeCell<*mut u8>,
    /// An optional destructor for this fragment
    destructor: Option<NonNull<dyn FnMut(NonNull<u8>)>>,
}
impl fmt::Debug for HeapFragment {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("HeapFragment")
            .field("raw", &self.raw)
            .field("top", &format_args!("{:p}", self.top.get()))
            .field(
                "destructor",
                &format_args!("{:?}", self.destructor.map(|ptr| ptr.as_ptr())),
            )
            .finish()
    }
}
impl HeapFragment {
    /// Returns the pointer to the data region of this fragment
    #[inline]
    pub fn data(&self) -> NonNull<u8> {
        self.raw.data()
    }

    /// Creates a new heap fragment with the given layout, allocated on the global heap
    #[inline]
    pub fn new(
        layout: Layout,
        destructor: Option<Box<dyn FnMut(NonNull<u8>)>>,
    ) -> Result<NonNull<Self>, AllocError> {
        let align = cmp::max(MIN_ALIGN, layout.align());
        let layout = layout.align_to(align).unwrap().pad_to_align();

        let (full_layout, offset) = Layout::new::<Self>().extend(layout.clone()).unwrap();
        let ptr: NonNull<u8> = Global.allocate(full_layout)?.cast();
        let header = ptr.as_ptr() as *mut Self;
        let base = unsafe { NonNull::new_unchecked(ptr.as_ptr().add(offset)) };
        unsafe {
            header.write(Self {
                link: LinkedListLink::new(),
                raw: RawFragment { layout, base },
                top: UnsafeCell::new(base.as_ptr()),
                destructor: destructor.map(|d| NonNull::new_unchecked(Box::into_raw(d))),
            });
            Ok(NonNull::new_unchecked(header))
        }
    }

    /// Sets the destructor for this fragment after it was constructed
    ///
    /// This function will panic if there is already a destructor set. It is intended
    /// for cases in which the fragment is known to not have a destructor, and we want
    /// to install one.
    pub unsafe fn set_destructor<F>(&mut self, destructor: F)
    where
        F: FnMut(NonNull<u8>) + 'static,
    {
        assert!(self.destructor.is_none());

        let destructor = Box::new(destructor);
        self.destructor = Some(unsafe { NonNull::new_unchecked(Box::into_raw(destructor)) });
    }
}
impl Drop for HeapFragment {
    fn drop(&mut self) {
        assert!(!self.link.is_linked());
        // Check if this fragment needs to have a destructor run
        if let Some(mut destructor) = self.destructor.take() {
            let dtor = unsafe { destructor.as_mut() };
            dtor(self.raw.base);
            unsafe {
                ptr::drop_in_place(dtor);
            }
        }

        // Deallocate the memory backing this fragment
        let (layout, _offset) = Layout::new::<Self>().extend(self.raw.layout()).unwrap();
        unsafe {
            let ptr = NonNull::new_unchecked(self as *const _ as *mut u8);

            Global.deallocate(ptr, layout);
        }
    }
}
unsafe impl Allocator for HeapFragment {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let layout = layout.pad_to_align();
        let size = layout.size();

        // Calculate the base pointer of the allocation at the desired alignment,
        // then offset that pointer by the desired size to give us the new top
        let top = unsafe { &mut *self.top.get() };
        let offset = top.align_offset(layout.align());
        let base = unsafe { top.add(offset) };
        let new_top = unsafe { base.add(size) };

        // Make sure the requested allocation fits within the fragment
        let range = self.raw.as_ptr_range();
        if range.contains(&new_top) || range.end == new_top {
            *top = new_top;
            Ok(unsafe { NonNull::new_unchecked(ptr::from_raw_parts_mut(base.cast(), size)) })
        } else {
            Err(AllocError)
        }
    }

    /// Deallocation is ignored unless the region referenced is at the top of the heap
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let ptr = ptr.as_ptr();
        let end = unsafe { ptr.add(layout.size()) };
        let top = unsafe { &mut *self.top.get() };
        if end < *top {
            return;
        }
        *top = ptr;
    }
    // The following functions are all no-ops or errors with heap fragments
    unsafe fn grow(
        &self,
        _ptr: NonNull<u8>,
        _old_layout: Layout,
        _new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        Err(AllocError)
    }
    unsafe fn grow_zeroed(
        &self,
        _ptr: NonNull<u8>,
        _old_layout: Layout,
        _new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        Err(AllocError)
    }
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        _old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        Ok(NonNull::slice_from_raw_parts(ptr, new_layout.size()))
    }
}
impl Heap for HeapFragment {
    #[inline]
    fn heap_start(&self) -> *mut u8 {
        self.raw.base.as_ptr()
    }

    #[inline]
    fn heap_top(&self) -> *mut u8 {
        unsafe { *self.top.get() }
    }

    #[inline]
    unsafe fn reset_heap_top(&self, top: *mut u8) {
        unsafe {
            *self.top.get() = top;
        }
    }

    #[inline]
    fn heap_end(&self) -> *mut u8 {
        self.raw.as_ptr_range().end
    }
}

#[cfg(test)]
mod tests {
    use core::sync::atomic::{AtomicU8, Ordering};

    use super::*;

    #[test]
    fn heap_fragment_integration_test() {
        static COUNTER: AtomicU8 = AtomicU8::new(0);

        struct Drops(usize);
        impl Drop for Drops {
            fn drop(&mut self) {
                COUNTER.fetch_add(self.0.try_into().unwrap(), Ordering::Relaxed);
            }
        }

        let layout = Layout::new::<[Drops; 3]>();
        let destructor = Box::new(|ptr: NonNull<u8>| {
            let ptr = ptr.cast::<[Drops; 3]>();
            unsafe {
                core::ptr::drop_in_place(ptr.as_ptr());
            }
        });
        let fragment_ptr = HeapFragment::new(layout, Some(destructor)).unwrap();
        let fragment = unsafe { fragment_ptr.as_ref() };
        let data_ptr: NonNull<u8> = fragment.allocate(layout).unwrap().cast();
        unsafe {
            let array_ptr = data_ptr.cast::<[Drops; 3]>().as_ptr();
            array_ptr.write([Drops(1), Drops(2), Drops(3)]);
            core::ptr::drop_in_place(fragment_ptr.as_ptr());
        }

        let count = COUNTER.load(Ordering::Relaxed);
        assert_eq!(count, 6);
    }

    #[test]
    fn heap_fragment_no_destructor_integration_test() {
        static COUNTER: AtomicU8 = AtomicU8::new(0);

        struct Drops(usize);
        impl Drop for Drops {
            fn drop(&mut self) {
                COUNTER.fetch_add(self.0.try_into().unwrap(), Ordering::Relaxed);
            }
        }

        let layout = Layout::new::<[Drops; 3]>();
        let fragment_ptr = HeapFragment::new(layout, None).unwrap();
        let fragment = unsafe { fragment_ptr.as_ref() };
        let data_ptr: NonNull<u8> = fragment.allocate(layout).unwrap().cast();
        unsafe {
            let array_ptr = data_ptr.cast::<[Drops; 3]>().as_ptr();
            array_ptr.write([Drops(1), Drops(2), Drops(3)]);
            core::ptr::drop_in_place(fragment_ptr.as_ptr());
        }

        let count = COUNTER.load(Ordering::Relaxed);
        assert_eq!(count, 0);
    }
}
