use core::alloc::Layout;
use core::ptr::NonNull;

use crate::erts::exception::AllocResult;
use crate::erts::process::alloc::*;
use crate::erts::term::prelude::*;

use super::{Generation, Sweep, Sweepable, Sweeper};

/// Represents a collection algorithm that sweeps for references in `Target`
/// into `Source`, and moving them into `Target`
pub trait CollectionType: HeapAlloc {
    type Source: Heap + VirtualAlloc;
    type Target: Heap + VirtualAlloc;

    // Obtain immutable reference to underlying source heap
    fn source(&self) -> &Self::Source;

    // Obtain mutable reference to underlying source heap
    fn source_mut(&self) -> &mut Self::Source;

    // Obtain immutable reference to underlying target heap
    fn target(&self) -> &Self::Target;

    // Obtain mutable reference to underlying target heap
    fn target_mut(&self) -> &mut Self::Target;

    /// Performs a collection using an instance of this type
    fn collect(&mut self) -> usize;
}

/// An implementation of `CollectionType` for full-sweep collections, where
/// references in the target to either generation in the old heap, are swept
/// into the new heap represented by `target`. It is expected that the root
/// set has already been swept into `target`
pub struct FullCollection<'a, S, T>
where
    S: GenerationalHeap,
    T: Heap,
{
    source: &'a mut S,
    target: &'a mut T,
}
impl<'a, S, T> FullCollection<'a, S, T>
where
    S: GenerationalHeap,
    T: Heap + VirtualAlloc,
{
    pub fn new(source: &'a mut S, target: &'a mut T) -> Self {
        Self { source, target }
    }
}
impl<'a, S, T> HeapAlloc for FullCollection<'a, S, T>
where
    S: GenerationalHeap,
    T: Heap + VirtualAlloc,
{
    #[inline]
    unsafe fn alloc_layout(&mut self, layout: Layout) -> AllocResult<NonNull<Term>> {
        self.target.alloc_layout(layout)
    }
}
impl<'a, S, T> CollectionType for FullCollection<'a, S, T>
where
    S: GenerationalHeap,
    T: Heap + VirtualAlloc,
{
    type Source = S;
    type Target = T;

    fn source(&self) -> &Self::Source {
        self.source
    }

    fn source_mut(&self) -> &mut Self::Source {
        unsafe { &mut *(self.source as *const S as *mut S) }
    }

    fn target(&self) -> &Self::Target {
        self.target
    }

    fn target_mut(&self) -> &mut Self::Target {
        unsafe { &mut *(self.target as *const T as *mut T) }
    }

    fn collect(&mut self) -> usize {
        let mut moved = 0;
        for term in self.target.iter_mut() {
            moved += unsafe { sweep_term(self, term) };
        }
        moved
    }
}

/// An implementation of `CollectionType` for minor collections, where
/// references in the young generation of `target` to `source` are swept
/// into either the old generation or young generation of `target`, depending
/// on object maturity. It is expected that the root set has already been swept
/// into the young generation of `target`.
pub struct MinorCollection<'a, S, T>
where
    S: Heap,
    T: GenerationalHeap,
{
    source: &'a mut S,
    target: &'a mut T,
    mode: Generation,
}
impl<'a, S, T> MinorCollection<'a, S, T>
where
    S: Heap + VirtualAlloc,
    T: GenerationalHeap,
{
    pub fn new(source: &'a mut S, target: &'a mut T) -> Self {
        Self {
            source,
            target,
            mode: Generation::Young,
        }
    }

    /// Determine the generation to move the given pointer to
    ///
    /// If `None`, then no move is required
    fn get_generation<P: ?Sized>(&self, ptr: *mut P) -> Option<Generation> {
        use liblumen_core::util::pointer::in_area;
        // In a minor collection, we move mature objects into the old generation,
        // otherwise they are moved into the young generation. Objects already in
        // the young/old generation do not need to be moved
        if self.target.contains(ptr) {
            return None;
        }

        // Checking maturity to select destination
        if in_area(ptr, self.source.heap_start(), self.source.high_water_mark()) {
            return Some(Generation::Old);
        }

        // If the object isn't in the mature region, and isn't in the target, then
        // it must be moved into the young generation
        Some(Generation::Young)
    }
}
impl<'a, S, T> HeapAlloc for MinorCollection<'a, S, T>
where
    S: Heap + VirtualAlloc,
    T: GenerationalHeap,
{
    #[inline]
    unsafe fn alloc_layout(&mut self, layout: Layout) -> AllocResult<NonNull<Term>> {
        match self.mode {
            Generation::Young => self.target.young_generation_mut().alloc_layout(layout),
            Generation::Old => self.target.old_generation_mut().alloc_layout(layout),
        }
    }
}
impl<'a, S, T> CollectionType for MinorCollection<'a, S, T>
where
    S: Heap + VirtualAlloc,
    T: GenerationalHeap,
{
    type Source = S;
    type Target = T;

    fn source(&self) -> &Self::Source {
        self.source
    }

    fn source_mut(&self) -> &mut Self::Source {
        unsafe { &mut *(self.source as *const S as *mut S) }
    }

    fn target(&self) -> &Self::Target {
        self.target
    }

    fn target_mut(&self) -> &mut Self::Target {
        unsafe { &mut *(self.target as *const T as *mut T) }
    }

    fn collect(&mut self) -> usize {
        let mut moved = 0;
        let young = self.target.young_generation_mut();
        for term in young.iter_mut() {
            moved += unsafe { sweep_term(self, term) };
        }
        moved
    }
}
impl<'a, S, T> Sweeper for MinorCollection<'a, S, T>
where
    S: Heap + VirtualAlloc,
    T: GenerationalHeap,
{
    /// To avoid a redundant check, we always return true here,
    /// and then use our opportunity to check based on generation
    /// in the implementation of `sweep`
    #[inline(always)]
    fn should_sweep(&self, _raw: *mut Term) -> bool {
        true
    }
}
unsafe impl<'a, S, T, P> Sweep<P> for MinorCollection<'a, S, T>
where
    S: Heap + VirtualAlloc,
    T: GenerationalHeap,
    P: Sweepable<Self>,
{
    #[inline]
    unsafe fn sweep(&mut self, ptr: P) -> Option<(*mut Term, usize)> {
        match self.get_generation(ptr.into()) {
            Some(mode) => {
                let prev_mode = self.mode;
                self.mode = mode;
                let result = P::sweep(ptr, self);
                self.mode = prev_mode;
                Some(result)
            }
            None => None,
        }
    }
}

/// Collect all references from `Target` into `Source` by moving the
/// referenced values into `Target`. This is essentially a full collection,
/// but more general as it doesn't assume that the source is a generational
/// heap
pub struct ReferenceCollection<'a, S, T> {
    source: &'a mut S,
    target: &'a mut T,
}
impl<'a, S, T> ReferenceCollection<'a, S, T>
where
    S: Heap + VirtualAlloc,
    T: Heap + VirtualAlloc,
{
    pub fn new(source: &'a mut S, target: &'a mut T) -> Self {
        Self { source, target }
    }
}
impl<'a, S, T> HeapAlloc for ReferenceCollection<'a, S, T>
where
    S: Heap + VirtualAlloc,
    T: Heap + VirtualAlloc,
{
    #[inline]
    unsafe fn alloc_layout(&mut self, layout: Layout) -> AllocResult<NonNull<Term>> {
        self.target.alloc_layout(layout)
    }
}
impl<'a, S, T> CollectionType for ReferenceCollection<'a, S, T>
where
    S: Heap + VirtualAlloc,
    T: Heap + VirtualAlloc,
{
    type Source = S;
    type Target = T;

    fn source(&self) -> &Self::Source {
        self.source
    }

    fn source_mut(&self) -> &mut Self::Source {
        unsafe { &mut *(self.source as *const S as *mut S) }
    }

    fn target(&self) -> &Self::Target {
        self.target
    }

    fn target_mut(&self) -> &mut Self::Target {
        unsafe { &mut *(self.target as *const T as *mut T) }
    }

    fn collect(&mut self) -> usize {
        let mut moved = 0;
        for term in self.target.iter_mut() {
            moved += unsafe { sweep_term(self, term) };
        }
        moved
    }
}

// This function contains the sweeping logic applied to terms in
// the root set. These are different than the terms swept by `sweep_term`
// in that they can be pointers to pointers, so we need to sweep the inner
// term and also update the root itself in those cases.
//
// Internally this delegates to `sweep_term` to do the sweep of the inner value
pub(super) unsafe fn sweep_root<G>(sweeper: &mut G, term: &mut Term) -> usize
where
    G: Sweeper
        + Sweep<*mut Term>
        + Sweep<Boxed<ProcBin>>
        + Sweep<Boxed<SubBinary>>
        + Sweep<Boxed<MatchContext>>
        + Sweep<Boxed<Cons>>,
{
    let pos = term as *mut Term;

    // We're sweeping roots, these should all be pointers
    debug_assert!(term.is_boxed() || term.is_non_empty_list());

    if term.is_boxed() {
        let term_ptr: *mut Term = term.dyn_cast();
        let inner_term = &mut *term_ptr;

        // If this is a pointer to a pointer, delegate to sweep_term,
        // then update the original root pointer
        if inner_term.is_boxed() {
            let moved = sweep_term(sweeper, inner_term);

            // Fetch forwarding address from marker
            let new_term = *term_ptr;
            debug_assert!(new_term.is_boxed());

            // Update root
            pos.write(new_term);

            moved
        } else if inner_term.is_non_empty_list() {
            let moved = sweep_term(sweeper, inner_term);

            // Fetch forwarding address from marker
            let marker = &*(term_ptr as *mut Cons);
            let new_term = marker.tail;
            debug_assert!(new_term.is_non_empty_list());

            // Update root
            pos.write(new_term);

            moved
        } else {
            assert!(inner_term.is_header());
            // This is just a pointer, not a pointer to a pointer,
            // so sweep it normally
            sweep_term(sweeper, term)
        }
    } else {
        assert!(term.is_non_empty_list());
        // This is just a list pointer, not a pointer to a list pointer,
        // so sweep it normally
        sweep_term(sweeper, term)
    }
}

// This function contains the generic default sweeping logic applied against
// a given term on the heap, using the provided context to execute any
// required moves.
//
// This is invoked by collectors/collection types to kick off the sweep of
// a single term. Once the type of the term is determined, type-specific sweeps
// are handled by implementations of the `Sweepable` trait.
pub(super) unsafe fn sweep_term<G>(sweeper: &mut G, term: &mut Term) -> usize
where
    G: Sweeper
        + Sweep<*mut Term>
        + Sweep<Boxed<ProcBin>>
        + Sweep<Boxed<SubBinary>>
        + Sweep<Boxed<MatchContext>>
        + Sweep<Boxed<Cons>>,
{
    let pos = term as *mut Term;

    if term.is_boxed() {
        // Skip pointers to literals
        if term.is_literal() {
            return 0;
        }

        // Check if this is a move marker
        let box_ptr: *mut Term = (*pos).dyn_cast();
        let unboxed = &*box_ptr;
        if unboxed.is_boxed() {
            // Overwrite the move marker with the forwarding address
            pos.write(*unboxed);
            return 0;
        }

        assert!(unboxed.is_header());

        if unboxed.is_procbin() {
            let bin: Boxed<ProcBin> = Boxed::new_unchecked(box_ptr as *mut ProcBin);
            if let Some((new_ptr, moved)) = sweeper.sweep(bin) {
                // Write marker
                let marker: Term = new_ptr.into();
                pos.write(marker);
                return moved;
            }
            return 0;
        }

        if unboxed.is_subbinary() {
            let bin: Boxed<SubBinary> = Boxed::new_unchecked(box_ptr as *mut SubBinary);
            if let Some((new_ptr, moved)) = sweeper.sweep(bin) {
                // Write marker
                let marker: Term = new_ptr.into();
                pos.write(marker);
                return moved;
            }
            return 0;
        }

        if let Some((new_ptr, moved)) = sweeper.sweep(box_ptr) {
            let marker: Term = new_ptr.into();
            pos.write(marker);
            return moved;
        }

        return 0;
    }

    if term.is_non_empty_list() {
        // Skip pointers to literals
        if term.is_literal() {
            return 0;
        }

        // Check if this is a move marker
        let ptr: Boxed<Cons> = (*pos).dyn_cast();
        let cons = ptr.as_ref();
        if cons.is_move_marker() {
            // Overwrite the move marker with the forwarding address
            pos.write(cons.tail);
            return 0;
        }

        // Move the pointed-to value
        if let Some((new_ptr, moved)) = sweeper.sweep(ptr) {
            // For cons cells only, we need to make sure we encode the pointer as a list type
            let marker: Term = (new_ptr as *mut Cons).into();
            pos.write(marker);
            return moved;
        }
        return 0;
    }

    // When we encounter a header for a match context, check if we should also move its referenced
    // binary
    if term.is_match_context() {
        let bin: Boxed<MatchContext> = Boxed::new_unchecked(pos as *mut MatchContext);
        if let Some((new_ptr, moved)) = sweeper.sweep(bin) {
            let marker: Term = new_ptr.into();
            pos.write(marker);
            return moved;
        }
        return 0;
    }

    // For all other headers, perform a simple move
    if let Some((new_ptr, moved)) = sweeper.sweep(pos) {
        let marker: Term = new_ptr.into();
        pos.write(marker);
        return moved;
    }

    return 0;
}
