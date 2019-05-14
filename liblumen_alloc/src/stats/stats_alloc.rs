use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering;
use core::intrinsics::type_name;
use core::ptr::{self, NonNull};
use core::alloc::{GlobalAlloc, Alloc, AllocErr, Layout};
use core::fmt;

use liblumen_core::locks::SpinLock;

use super::hooks;
use super::histogram::Histogram;

const NUM_SIZE_CLASSES: u64 = 67;

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
pub struct StatsAlloc<T> {
    alloc_calls: AtomicUsize,
    dealloc_calls: AtomicUsize,
    realloc_calls: AtomicUsize,
    total_bytes_alloced: AtomicUsize,
    total_bytes_freed: AtomicUsize,

    tag: &'static str,
    histogram: SpinLock<Histogram>,
    allocator: T,
}
impl<T> StatsAlloc<T> {
    #[inline]
    pub fn new(t: T) -> Self {
        Self {
            alloc_calls: AtomicUsize::new(0),
            dealloc_calls: AtomicUsize::new(0),
            realloc_calls: AtomicUsize::new(0),
            total_bytes_alloced: AtomicUsize::new(0),
            total_bytes_freed: AtomicUsize::new(0),
            tag: unsafe { type_name::<T>() },
            histogram: SpinLock::new(Histogram::with_buckets(NUM_SIZE_CLASSES)),
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
            tag,
            histogram: SpinLock::new(Histogram::with_buckets(NUM_SIZE_CLASSES)),
            allocator: t,
        }
    }

    #[inline]
    pub fn stats(&self) -> Statistics {
        let h = self.histogram.lock();
        let histogram = h.clone();
        drop(h);
        Statistics {
            alloc_calls: self.alloc_calls.load(Ordering::Relaxed),
            realloc_calls: self.realloc_calls.load(Ordering::Relaxed),
            dealloc_calls: self.dealloc_calls.load(Ordering::Relaxed),
            total_bytes_alloced: self.total_bytes_alloced.load(Ordering::Relaxed),
            total_bytes_freed: self.total_bytes_freed.load(Ordering::Relaxed),
            tag: self.tag,
            histogram,
        }
    }
}
impl<T: Default> Default for StatsAlloc<T> {
    #[inline]
    fn default() -> Self {
        Self::new(T::default())
    }
}
unsafe impl<T: Alloc + Sync> Sync for StatsAlloc<T> {}
unsafe impl<T: Alloc + Send> Send for StatsAlloc<T> {}


/// This struct represents a snapshot of the stats gathered
/// by an instances of `StatsAlloc`, and is used for display
#[derive(Debug)]
pub struct Statistics {
    alloc_calls: usize,
    dealloc_calls: usize,
    realloc_calls: usize,
    total_bytes_alloced: usize,
    total_bytes_freed: usize,
    
    tag: &'static str,
    histogram: Histogram,
}
impl fmt::Display for Statistics {
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

unsafe impl<T: Alloc> Alloc for StatsAlloc<T> {
    #[inline]
    unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocErr> {
        let size = layout.size();
        let align = layout.align();
        match self.allocator.alloc(layout) {
            err @ Err(_) => {
                hooks::on_alloc(self.tag, size, align, ptr::null_mut());
                err
            },
            Ok(result) => {
                self.alloc_calls.fetch_add(1, Ordering::SeqCst);
                self.total_bytes_alloced.fetch_add(size, Ordering::SeqCst);
                let mut h = self.histogram.lock();
                h.add(size as u64);
                drop(h);
                hooks::on_alloc(self.tag, size, align, result.as_ptr());
                Ok(result)
            }
        }
    }

    #[inline]
    unsafe fn realloc(&mut self, ptr: NonNull<u8>, layout: Layout, new_size: usize) -> Result<NonNull<u8>, AllocErr> {
        let old_ptr = ptr.as_ptr();
        let old_size = layout.size();
        let align = layout.align();
        match self.allocator.realloc(ptr, layout, new_size) {
            err @ Err(_) => {
                hooks::on_realloc(self.tag, old_size, new_size, align, old_ptr, ptr::null_mut());
                err
            },
            Ok(result) => {
                self.realloc_calls.fetch_add(1, Ordering::SeqCst);
                if old_size < new_size {
                    let diff = new_size - old_size;
                    self.total_bytes_alloced.fetch_add(diff, Ordering::SeqCst);
                } else {
                    let diff = old_size - new_size;
                    self.total_bytes_alloced.fetch_sub(diff, Ordering::SeqCst);
                }
                let mut h = self.histogram.lock();
                h.add(new_size as u64);
                drop(h);
                hooks::on_realloc(self.tag, old_size, new_size, align, old_ptr, result.as_ptr());
                Ok(result)
            }
        }
    }

    #[inline]
    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        let size = layout.size();
        let align = layout.align();
        let freed = ptr.as_ptr();
        self.allocator.dealloc(ptr, layout);
        self.dealloc_calls.fetch_add(1, Ordering::SeqCst);
        self.total_bytes_freed.fetch_add(size, Ordering::SeqCst);
        hooks::on_dealloc(self.tag, size, align, freed);
    }
}

unsafe impl<T: GlobalAlloc> GlobalAlloc for StatsAlloc<T> {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        let align = layout.align();
        let result = self.allocator.alloc(layout);
        if result.is_null() {
            hooks::on_alloc(self.tag, size, align, result);
            return result;
        }

        self.alloc_calls.fetch_add(1, Ordering::SeqCst);
        self.total_bytes_alloced.fetch_add(size, Ordering::SeqCst);
        let mut h = self.histogram.lock();
        h.add(size as u64);
        drop(h);
        hooks::on_alloc(self.tag, size, align, result);

        result
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let old_size = layout.size();
        let align = layout.align();

        let result = self.allocator.realloc(ptr, layout, new_size);
        if result.is_null() {
            hooks::on_realloc(self.tag, old_size, new_size, align, ptr, result);
            return result;
        }

        self.realloc_calls.fetch_add(1, Ordering::SeqCst);
        if old_size < new_size {
            let diff = new_size - old_size;
            self.total_bytes_alloced.fetch_add(diff, Ordering::SeqCst);
        } else {
            let diff = old_size - new_size;
            self.total_bytes_alloced.fetch_sub(diff, Ordering::SeqCst);
        }
        let mut h = self.histogram.lock();
        h.add(new_size as u64);
        drop(h);
        hooks::on_realloc(self.tag, old_size, new_size, align, ptr, result);

        result
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let size = layout.size();
        let align = layout.align();
        self.allocator.dealloc(ptr, layout);
        self.dealloc_calls.fetch_add(1, Ordering::SeqCst);
        self.total_bytes_freed.fetch_add(size, Ordering::SeqCst);
        hooks::on_dealloc(self.tag, size, align, ptr);
    }
}