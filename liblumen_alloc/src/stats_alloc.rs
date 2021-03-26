use core::fmt;
use core::intrinsics::type_name;
use core::ptr::{self, NonNull};
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering;

use liblumen_core::alloc::prelude::*;
use liblumen_core::locks::RwLock;

use crate::stats::hooks;
use crate::stats::{DefaultHistogram, Histogram};

/// `StatsAlloc` is a tracing allocator which wraps some
/// allocator, either an implementation of `Alloc` or `GlobalAlloc`,
/// and tracks statistics about usage of that allocator:
///
/// - The number of calls to alloc/realloc/dealloc
/// - The total number of bytes allocated and freed
/// - A histogram of allocation sizes
///
/// The `StatsAlloc` can be tagged to provide useful metadata bout
/// what type of allocator is being traced and how it is used.
#[derive(Debug)]
pub struct StatsAlloc<T, H: Histogram + Clone + Default = DefaultHistogram> {
    alloc_calls: AtomicUsize,
    dealloc_calls: AtomicUsize,
    realloc_calls: AtomicUsize,
    total_bytes_alloced: AtomicUsize,
    total_bytes_freed: AtomicUsize,
    histogram: RwLock<H>,

    tag: &'static str,
    allocator: T,
}
impl<T, H: Histogram + Clone + Default> StatsAlloc<T, H> {
    #[inline]
    pub fn new(t: T) -> Self {
        Self {
            alloc_calls: AtomicUsize::new(0),
            dealloc_calls: AtomicUsize::new(0),
            realloc_calls: AtomicUsize::new(0),
            total_bytes_alloced: AtomicUsize::new(0),
            total_bytes_freed: AtomicUsize::new(0),
            histogram: RwLock::default(),
            tag: type_name::<T>(),
            allocator: t,
        }
    }

    #[inline]
    pub fn new_tagged(t: T, tag: &'static str) -> Self {
        Self {
            alloc_calls: AtomicUsize::new(0),
            dealloc_calls: AtomicUsize::new(0),
            realloc_calls: AtomicUsize::new(0),
            total_bytes_alloced: AtomicUsize::new(0),
            total_bytes_freed: AtomicUsize::new(0),
            histogram: RwLock::default(),
            tag,
            allocator: t,
        }
    }

    #[inline]
    pub fn stats(&self) -> Statistics<H> {
        let h = self.histogram.read();
        let histogram = h.clone();
        drop(h);
        Statistics {
            alloc_calls: self.alloc_calls.load(Ordering::Relaxed),
            realloc_calls: self.realloc_calls.load(Ordering::Relaxed),
            dealloc_calls: self.dealloc_calls.load(Ordering::Relaxed),
            total_bytes_alloced: self.total_bytes_alloced.load(Ordering::Relaxed),
            total_bytes_freed: self.total_bytes_freed.load(Ordering::Relaxed),
            histogram,
            tag: self.tag,
        }
    }
}
impl<T: Default, H: Histogram + Clone + Default> Default for StatsAlloc<T, H> {
    #[inline]
    fn default() -> Self {
        Self::new(T::default())
    }
}
unsafe impl<T: Allocator + Sync, H: Histogram + Clone + Default> Sync for StatsAlloc<T, H> {}
unsafe impl<T: Allocator + Send, H: Histogram + Clone + Default> Send for StatsAlloc<T, H> {}

/// This struct represents a snapshot of the stats gathered
/// by an instances of `StatsAlloc`, and is used for display
#[derive(Debug)]
pub struct Statistics<H: Histogram + Clone + Default> {
    alloc_calls: usize,
    dealloc_calls: usize,
    realloc_calls: usize,
    total_bytes_alloced: usize,
    total_bytes_freed: usize,

    tag: &'static str,
    histogram: H,
}
impl<H: Histogram + Clone + Default> fmt::Display for Statistics<H> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "## Allocator Statistics (tag = {})", self.tag)?;
        writeln!(f, "# Calls to alloc = {}", self.alloc_calls)?;
        writeln!(f, "# Calls to realloc = {}", self.realloc_calls)?;
        writeln!(f, "# Calls to dealloc = {}", self.dealloc_calls)?;
        writeln!(f, "#")?;
        writeln!(f, "# Total Bytes Allocated = {}", self.total_bytes_alloced)?;
        writeln!(f, "# Total Bytes Freed = {}", self.total_bytes_freed)?;
        writeln!(f, "#")?;
        writeln!(f, "# Allocations Histogram:")?;
        writeln!(f, "{}", self.histogram)
    }
}

unsafe impl<T: Allocator, H: Histogram + Clone + Default> Allocator for StatsAlloc<T, H> {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let size = layout.size();
        let align = layout.align();
        match self.allocator.allocate(layout) {
            err @ Err(_) => {
                hooks::on_alloc(self.tag.to_owned(), size, align, ptr::null_mut());
                err
            }
            Ok(non_null_byte_slice) => {
                self.alloc_calls.fetch_add(1, Ordering::SeqCst);
                self.total_bytes_alloced.fetch_add(size, Ordering::SeqCst);
                let mut h = self.histogram.write();
                h.add(size as u64).ok();
                drop(h);
                hooks::on_alloc(
                    self.tag.to_owned(),
                    size,
                    align,
                    non_null_byte_slice.as_mut_ptr(),
                );
                Ok(non_null_byte_slice)
            }
        }
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let size = layout.size();
        let align = layout.align();
        let freed = ptr.as_ptr();
        self.allocator.deallocate(ptr, layout);
        self.dealloc_calls.fetch_add(1, Ordering::SeqCst);
        self.total_bytes_freed.fetch_add(size, Ordering::SeqCst);
        hooks::on_dealloc(self.tag.to_owned(), size, align, freed);
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let old_ptr = ptr.as_ptr();
        let old_size = old_layout.size();
        let old_align = old_layout.align();
        let new_size = new_layout.size();
        let new_align = new_layout.align();
        match self.allocator.grow(ptr, old_layout, new_layout) {
            err @ Err(_) => {
                hooks::on_realloc(
                    self.tag.to_owned(),
                    old_size,
                    new_size,
                    old_align,
                    new_align,
                    old_ptr,
                    ptr::null_mut(),
                );
                err
            }
            Ok(non_null_byte_slice) => {
                self.realloc_calls.fetch_add(1, Ordering::SeqCst);
                let diff = new_size - old_size;
                self.total_bytes_alloced.fetch_add(diff, Ordering::SeqCst);
                let mut h = self.histogram.write();
                h.add(new_size as u64).ok();
                drop(h);
                hooks::on_realloc(
                    self.tag.to_owned(),
                    old_size,
                    new_size,
                    old_align,
                    new_align,
                    old_ptr,
                    non_null_byte_slice.as_mut_ptr(),
                );
                Ok(non_null_byte_slice)
            }
        }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let old_ptr = ptr.as_ptr();
        let old_size = old_layout.size();
        let old_align = old_layout.align();
        let new_size = new_layout.size();
        let new_align = new_layout.align();
        match self.allocator.shrink(ptr, old_layout, new_layout) {
            err @ Err(_) => {
                hooks::on_realloc(
                    self.tag.to_owned(),
                    old_size,
                    new_size,
                    old_align,
                    new_align,
                    old_ptr,
                    ptr::null_mut(),
                );
                err
            }
            Ok(non_null_byte_slice) => {
                self.realloc_calls.fetch_add(1, Ordering::SeqCst);
                let diff = old_size - non_null_byte_slice.len();
                self.total_bytes_alloced.fetch_sub(diff, Ordering::SeqCst);
                let mut h = self.histogram.write();
                h.add(non_null_byte_slice.len() as u64).ok();
                drop(h);
                hooks::on_realloc(
                    self.tag.to_owned(),
                    old_size,
                    new_size,
                    old_align,
                    new_align,
                    old_ptr,
                    non_null_byte_slice.as_mut_ptr(),
                );
                Ok(non_null_byte_slice)
            }
        }
    }
}

unsafe impl<T: GlobalAlloc, H: Histogram + Clone + Default> GlobalAlloc for StatsAlloc<T, H> {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        let align = layout.align();
        let result = self.allocator.alloc(layout);
        if result.is_null() {
            hooks::on_alloc(self.tag.to_owned(), size, align, result);
            return result;
        }

        self.alloc_calls.fetch_add(1, Ordering::SeqCst);
        self.total_bytes_alloced.fetch_add(size, Ordering::SeqCst);
        let mut h = self.histogram.write();
        h.add(size as u64).ok();
        drop(h);
        hooks::on_alloc(self.tag.to_owned(), size, align, result);

        result
    }

    #[inline]
    unsafe fn realloc(&self, old_ptr: *mut u8, old_layout: Layout, new_size: usize) -> *mut u8 {
        let old_size = old_layout.size();
        let old_align = old_layout.align();
        let new_align = old_align;

        let new_ptr = self.allocator.realloc(old_ptr, old_layout, new_size);
        if new_ptr.is_null() {
            hooks::on_realloc(
                self.tag.to_owned(),
                old_size,
                new_size,
                old_align,
                new_align,
                old_ptr,
                new_ptr,
            );
            return new_ptr;
        }

        self.realloc_calls.fetch_add(1, Ordering::SeqCst);
        if old_size < new_size {
            let diff = new_size - old_size;
            self.total_bytes_alloced.fetch_add(diff, Ordering::SeqCst);
        } else {
            let diff = old_size - new_size;
            self.total_bytes_alloced.fetch_sub(diff, Ordering::SeqCst);
        }
        let mut h = self.histogram.write();
        h.add(new_size as u64).ok();
        drop(h);
        hooks::on_realloc(
            self.tag.to_owned(),
            old_size,
            new_size,
            old_align,
            new_align,
            old_ptr,
            new_ptr,
        );

        new_ptr
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let size = layout.size();
        let align = layout.align();
        self.allocator.dealloc(ptr, layout);
        self.dealloc_calls.fetch_add(1, Ordering::SeqCst);
        self.total_bytes_freed.fetch_add(size, Ordering::SeqCst);
        hooks::on_dealloc(self.tag.to_owned(), size, align, ptr);
    }
}
