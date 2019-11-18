use core::alloc::Layout;
use core::mem;
use core::ptr;

use crate::erts::process::alloc::*;
use crate::erts::term::prelude::*;

use super::CollectionType;

pub trait Sweeper: CollectionType {
    fn should_sweep(&self, raw: *mut Term) -> bool;
}

impl<G> Sweeper for G
where
    G: CollectionType,
{
    #[inline]
    default fn should_sweep(&self, raw: *mut Term) -> bool {
        self.target().contains(raw) == false
    }
}

pub unsafe trait Sweep<P>: Sweeper {
    unsafe fn sweep(&mut self, ptr: P) -> Option<(*mut Term, usize)>;
}

pub unsafe trait Sweepable<G: Sweeper>: Sized + Copy + Into<*mut Term> {
    unsafe fn sweep(self, sweeper: &mut G) -> (*mut Term, usize);
}

unsafe impl<G, P> Sweep<P> for G
where
    G: Sweeper,
    P: Sweepable<G>,
{
    #[inline]
    default unsafe fn sweep(&mut self, ptr: P) -> Option<(*mut Term, usize)> {
        if self.should_sweep(ptr.into()) {
            Some(P::sweep(ptr, self))
        } else {
            None
        }
    }
}

unsafe impl<G> Sweepable<G> for *mut Term
where
    G: Sweeper,
{
    unsafe fn sweep(self, sweeper: &mut G) -> (*mut Term, usize) {
        use crate::erts;
        use liblumen_core::sys::sysconf::MIN_ALIGN;

        let header = &*self;
        // The only context this should be used in is when moving a header
        assert!(header.is_header(), "invalid header {:?}", header);

        // Handle dynamically-sized types with large headers specially
        let size = if header.is_heapbin() {
            let bin = HeapBin::from_raw_term(self);
            mem::size_of_val(bin.as_ref())
        } else if header.is_function() {
            let closure = Closure::from_raw_term(self);
            mem::size_of_val(closure.as_ref())
        } else {
            header.sizeof()
        };

        // Round up to size in words
        let words = erts::to_word_size(size);

        let layout = Layout::from_size_align(words * mem::size_of::<Term>(), MIN_ALIGN)
            .unwrap()
            .pad_to_align()
            .unwrap();
        let total_size = layout.size();

        // Allocate space for move
        let dst = sweeper.alloc_layout(layout).unwrap().as_ptr();

        // Copy object to the destination, byte-wise
        ptr::copy_nonoverlapping(self as *const u8, dst as *mut u8, size);

        // Write move marker to previous location
        let marker: Term = dst.into();
        self.write(marker);

        (dst, total_size)
    }
}

// For cons cells, the move marker takes a different form than plain boxes.
// Cons cells are considered move markers when the head of the cell is the
// NONE value, and the tail is a boxed pointer to the cons cell which was
// moved.
unsafe impl<G> Sweepable<G> for Boxed<Cons>
where
    G: Sweeper,
{
    unsafe fn sweep(self, sweeper: &mut G) -> (*mut Term, usize) {
        let src = self.as_ptr();
        let layout = Layout::new::<Cons>();
        let size = layout.size();
        let dst = sweeper
            .alloc_layout(layout)
            .unwrap()
            .cast::<Cons>()
            .as_ptr();

        // Copy cons cell to new location
        src.copy_to_nonoverlapping(dst, 1);

        // Write move marker to previous location
        let marker: Term = dst.into();
        src.write(Cons::new(Term::NONE, marker));

        (dst as *mut Term, size)
    }
}

// Like cons cells, sub-binaries are handled a bit uniquely.
//
// While they use the same move marker strategy as other boxed values,
// sub-binaries hold weak references to other binaries on the heap,
// and we can't guarantee that the referred-to binary won't be collected,
// nor do we necessarily want to keep those binaries around if they are
// not needed.
//
// Instead, if the sub-binary points to a slice of the original binary
// that can be allocated with the heap binary limit (<= 64 bytes), then
// the sub-binary is promoted to a heap-binary, by copying the data from
// the original binary into a new heap binary allocation, rather than
// copying the sub-binary header.
//
// If the sub-binary is too large to fit in a heap binary allocation, then
// the original binary must be a reference-counted binary (procbin), so
// we don't need to worry about it being collected out from under us, as
// we count as a reference.
unsafe impl<G> Sweepable<G> for Boxed<SubBinary>
where
    G: Sweeper,
{
    unsafe fn sweep(self, sweeper: &mut G) -> (*mut Term, usize) {
        use crate::erts::string::Encoding;
        use core::slice;

        let sub = self.as_ref();
        let size = mem::size_of::<SubBinary>();

        // Determine if we should convert this sub-binary to a heapbin
        if let Ok((_bin_flags, bin_ptr, bin_size)) = sub.to_heapbin_parts() {
            // Allocate heap binary via copy
            let bytes = slice::from_raw_parts(bin_ptr, bin_size);
            let dst = HeapBin::from_slice(sweeper, bytes, Encoding::Raw).unwrap();
            // Get size of move
            let size = mem::size_of_val(dst.as_ref());

            (dst.cast::<Term>().as_ptr(), size)
        } else {
            // Move to new location
            let src = self.as_ptr();
            let dst = sweeper
                .alloc_layout(Layout::new::<SubBinary>())
                .unwrap()
                .cast::<SubBinary>()
                .as_ptr();
            src.copy_to_nonoverlapping(dst, 1);

            (dst as *mut Term, size)
        }
    }
}

// Match contexts are similar to sub-binaries, except rather than promoting the
// match context to a heap-allocated binary, we copy the original binary as part
// of moving the match context itself, except in those cases where the original
// binary is a reference-counted type
unsafe impl<G> Sweepable<G> for Boxed<MatchContext>
where
    G: Sweeper,
{
    unsafe fn sweep(self, sweeper: &mut G) -> (*mut Term, usize) {
        let ctx = self.as_ref();
        let src = self.as_ptr();
        let layout = Layout::new::<MatchContext>();
        let size = layout.size();
        let dst = sweeper
            .alloc_layout(layout)
            .unwrap()
            .cast::<MatchContext>()
            .as_ptr();

        let original_ptr: *mut Term = ctx.original().dyn_cast();
        let original = *original_ptr;

        // First, copy the context itself
        src.copy_to_nonoverlapping(dst, 1);

        // Acquire a reference to the moved match context for updating its pointers
        let mut moved_box: Boxed<MatchContext> = Boxed::new_unchecked(dst);
        let moved = moved_box.as_mut();

        // Next, move the referred to value if necessary

        // No move required for literals
        if original.is_literal() {
            return (dst as *mut Term, size);
        }

        // No move required for move markers, just need to update our reference
        if original.is_boxed() {
            let new_original_ptr: *mut Term = original.dyn_cast();
            let new_original = *new_original_ptr;
            let moved_original_ref = moved.original_mut();
            ptr::write(moved_original_ref, new_original_ptr.into());
            let moved_base_ref = moved.base_mut();
            ptr::write(moved_base_ref, new_original.as_binary_ptr());

            return (dst as *mut Term, size);
        }

        // We need to move the original reference
        let (new_original_ptr, bytes_moved) = if original.is_heapbin() {
            let bin = HeapBin::from_raw_term(original_ptr);
            bin.sweep(sweeper)
        } else if original.is_procbin() {
            let bin: Boxed<ProcBin> = original_ptr.into();
            bin.sweep(sweeper)
        } else {
            panic!(
                "encountered invalid binary reference in sub-binary: {:?}",
                original
            );
        };

        if new_original_ptr != original_ptr {
            // Original was moved, so update our reference to it
            let new_original = *new_original_ptr;
            let moved_original_ref = moved.original_mut();
            ptr::write(moved_original_ref, new_original_ptr.into());
            let moved_base_ref = moved.base_mut();
            ptr::write(moved_base_ref, new_original.as_binary_ptr());

            (dst as *mut Term, size + bytes_moved)
        } else {
            (dst as *mut Term, size)
        }
    }
}

unsafe impl<G> Sweepable<G> for Boxed<HeapBin>
where
    G: Sweeper,
{
    unsafe fn sweep(self, sweeper: &mut G) -> (*mut Term, usize) {
        // Copy to new region
        let bin = self.as_ref();
        let bytes = bin.as_bytes();
        let copy = HeapBin::from_slice(sweeper, bytes, bin.encoding()).unwrap();

        (
            copy.cast::<Term>().as_ptr(),
            mem::size_of_val(copy.as_ref()),
        )
    }
}

unsafe impl<G> Sweepable<G> for Boxed<ProcBin>
where
    G: Sweeper,
{
    unsafe fn sweep(self, sweeper: &mut G) -> (*mut Term, usize) {
        let src = self.as_ptr();

        // Allocate space for move
        let layout = Layout::new::<ProcBin>();
        let size = layout.size();
        let dst = sweeper
            .alloc_layout(layout)
            .unwrap()
            .cast::<ProcBin>()
            .as_ptr();

        let source = sweeper.source_mut();
        // Unlink from source virtual heap
        // This handles unlinking from either young or old generation
        // in those cases where the gc type is full and we're consolidating
        // into a new young generation
        if source.virtual_contains(src) {
            source.virtual_unlink(self);
        }

        // Move to new location
        src.copy_to_nonoverlapping(dst, 1);

        // Link to destination virtual heap
        let boxed = Boxed::new_unchecked(dst);
        sweeper.target_mut().virtual_alloc(boxed);

        (dst as *mut Term, size)
    }
}
