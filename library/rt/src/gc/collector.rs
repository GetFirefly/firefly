use core::mem;
use core::ops::Range;
use core::ptr;

use firefly_alloc::heap::{GenerationalHeap, Heap, HeapMut, SemispaceHeap};

use log::trace;

use crate::term::{BigInt, BinaryData, BitSlice, Closure, Pid, Reference, SmallMap, Tuple};
use crate::term::{Boxable, OpaqueTerm, Tag};

use super::*;

/// This is a bare bones collector which simple executes the
/// given sweep it is created with.
///
/// This is intended only for testing purposes to help isolate
/// issues in lower tiers of the GC infrastructure
pub struct SimpleCollector<T: CollectionType>(T);
impl<T> SimpleCollector<T>
where
    T: CollectionType,
{
    pub fn new(collection_type: T) -> Self {
        Self(collection_type)
    }
}
impl<'a> GarbageCollector for SimpleCollector<FullSweep<'a>> {
    /// Invokes the collector and uses the provided `need` (in words)
    /// to determine whether or not collection was successful/aggressive enough
    //
    // - Allocate to space
    // - Scan roots to find heap objects in from space to copy
    //      - Place move marker in previous heap position pointing to new heap position
    // - Scan to-space for pointers to from-space, and copy objects to to-space
    //      - NOTE: skip non-term values found on heap
    //      - Values copied here are allocated after `scan_stop` line, then `scan_stop` is adjusted
    //        past the object
    // - When `scan_stop` hits `scan_start`, we're done with the minor collection
    // - Deallocate from space
    fn garbage_collect(&mut self, roots: RootSet) -> Result<usize, GcError> {
        trace!(target: "process", "starting collection");

        // Follow roots and copy values to appropriate heap
        let moved = self.0.collect(roots)?;

        // Reap all dead references to ref-counted values on the old heap.
        //
        // NOTE: Any ref-counted references traceable from the roots have already been
        // cloned, so this is strictly a linear scan to drop unreachable references.
        //
        // NOTE: We do not scan the contents of moved containers, as the contents are
        // moved, not cloned, so we don't want to drop them. We're only scanning the
        // contents of dead containers.
        trace!(target: "process", "reaping dead references on old heap");
        self.0.source.reap();

        // Free the old generation heap, by swapping it out with a new empty
        // heap, resulting in it being dropped. The old generation is no
        // longer used post-sweep, until the next minor collection with a mature
        // region occurs
        trace!(target: "process", "dropping previous mature generation heap");
        self.0.source.swap_mature(ProcessHeap::empty());

        // Swap the target heap with the source young heap,
        // making the target the active heap
        //
        // A few things to note:
        //
        // - We use `swap` here because the new heap was initialized
        // outside of the collector, and when the collector returns,
        // that heap will be dropped when it falls out of scope. By
        // using `swap`, we are replacing the object that gets dropped
        // with the old young heap
        // - The reference given by `source.immature` is now the
        // new heap that was `target` prior to this swap
        trace!(target: "process", "swapping in target heap as new immature heap");
        mem::swap(self.0.source.immature_mut(), self.0.target);

        // Set the high water mark to the top of the immature heap,
        // which as mentioned above, is the heap to which we swept all
        // live values. On the next collection, the live values in the region
        // between the start of the immature heap and the top of the immature
        // heap will be those that get tenured into the mature region
        trace!(target: "process", "setting high water mark");
        let young = self.0.source.immature_mut();
        young.set_high_water_mark(young.heap_top());

        Ok(moved)
    }
}
impl<'h> GarbageCollector for SimpleCollector<MinorSweep<'h>> {
    fn garbage_collect(&mut self, roots: RootSet) -> Result<usize, GcError> {
        trace!(target: "process", "starting collection");

        // Track the top of the old generation to see if we promote any mature objects
        let old_top = self.0.target.mature().heap_top();

        // Follow roots and copy values to appropriate heap
        let mut moved = self.0.collect(roots)?;

        // Get mutable references to both generations
        let old = self.0.target.mature_mut();

        // If we have been tenuring (we have an old generation and have moved values into it),
        // then those newly tenured values may hold references into the old young generation
        // heap, which is about to be freed, so we need to move them into the old generation
        // as well (we never allow pointers into the young generation from the old)
        let has_tenured = old.heap_top() > old_top;
        if has_tenured {
            trace!(target: "process", "original heap has tenured objects, cleaning up..");
            let mut rc = ReferenceCollection::new(self.0.source, old);
            moved += rc.collect(RootSet::default())?;
        }

        // Mark where this collection ended in the new heap
        trace!(target: "process", "setting high water mark");
        let young = self.0.target.immature_mut();
        young.set_high_water_mark(young.heap_top());

        // Reap all dead references to ref-counted values on the old heap.
        trace!(target: "process", "reaping dead references on old heap");
        self.0.source.reap();

        Ok(moved)
    }
}

#[derive(Copy, Clone)]
pub struct HeapRange {
    start: *mut OpaqueTerm,
    end: *mut OpaqueTerm,
}
impl HeapRange {
    pub fn new(start: *mut OpaqueTerm, end: *mut OpaqueTerm) -> Self {
        assert!(start <= end);
        Self { start, end }
    }

    #[inline]
    pub fn skip(&mut self, n: usize) {
        let end = unsafe { self.start.add(n - 1) };
        if end >= self.end {
            self.start = self.end;
        } else {
            self.start = end;
        }
    }

    pub fn skip_bytes(&mut self, n: usize) {
        let end = unsafe { self.start.sub(1).byte_add(n) };
        if end >= self.end {
            self.start = self.end;
        } else {
            self.start = end;
        }
    }

    #[inline]
    pub fn peek(&self) -> Option<*mut OpaqueTerm> {
        let mut copy = *self;
        copy.next()
    }

    #[inline]
    pub fn next(&mut self) -> Option<*mut OpaqueTerm> {
        if self.start >= self.end {
            return None;
        }

        let mut start = self.start;
        // Always ensure the pointer is properly aligned
        if !start.is_aligned() {
            let offset = start
                .cast::<u8>()
                .align_offset(mem::align_of::<OpaqueTerm>());
            start = unsafe { start.byte_add(offset) };
        }
        unsafe {
            self.start = start.add(1);
        }
        Some(start)
    }
}

pub(crate) trait Reap {
    fn reap(&self);
}

impl<'a, A, B> Reap for SemispaceHeap<A, B>
where
    A: Heap + Reap,
    B: Heap + Reap,
{
    fn reap(&self) {
        self.immature().reap();
        self.mature().reap();
    }
}

impl Reap for ProcessHeap {
    #[inline]
    fn reap(&self) {
        self.used_range().reap();
    }
}

impl Reap for Range<*mut u8> {
    fn reap(&self) {
        let mut iter = HeapRange::new(
            self.start.cast::<OpaqueTerm>(),
            self.end.cast::<OpaqueTerm>(),
        );

        while let Some(ptr) = iter.next() {
            let term = unsafe { *ptr };

            // If we encounter a None followed by a cons pointer, this was a moved
            // cons cell and we skip over both head and tail.
            //
            // Otherwise we just skip over the None
            if term.is_none() {
                if let Some(next) = iter.peek() {
                    let term2 = unsafe { *next };
                    if term2.is_nonempty_list() {
                        iter.skip(1);
                    }
                    continue;
                } else {
                    // If we hit the end of the heap while performing this check, we're done
                    break;
                }
            }

            // A this point we have the following scenarios we can hit:
            //
            // 1. We visit the head term of a dead cons cell, we want to visit
            // these, but unless the head term is a ref-counted pointer, we can
            // skip over it to the tail term.
            // 2. We visit the tail term of a dead cons cell, same as above applies
            // 3. We visit the header of a moved term, which has been rewritten as
            // a forwarding pointer followed by a "hole" term indicating the size
            // of the moved value.
            // 4. Everything else can be ignored or is garbage
            //
            // We must check for 3 any time we encounter a pointer to a tuple or any
            // Gc<T> pointer, as it is not possible to distinguish 3 from 1 or 2 just
            // from the pointer term alone. All other term types can be safely treated
            // as 1 or 2.

            // A literal can always be skipped, and must be a head/tail term as described above
            if term.is_literal() {
                continue;
            }

            // If we encounter a NonNull<Tuple> or Gc<T> pointer followed by a Hole, this was
            // a moved heap-allocated term. The hole represents the size of the moved term,
            // so all we need to do is apply that size to the initial pointer to skip
            // over the moved value.
            if term.is_gcbox() || term.is_tuple() {
                if let Some(next) = iter.peek() {
                    let term2 = unsafe { *next };
                    if term2.is_hole() {
                        iter.skip_bytes(unsafe { term2.hole_size() });
                        continue;
                    }

                    // If we reach here, this pointer must be the head of a dead cons cell,
                    // so we can simply skip over it, as the next iteration will visit the tail.
                    continue;
                } else {
                    break;
                }
            }

            // If we reach a reference-counted pointer then we're in the head/tail of a dead cons cell
            if term.is_rc() {
                term.maybe_decrement_refcount();
                continue;
            }

            // All other term types can be skipped, and must be located in a dead cons cell
            if !term.is_header() {
                continue;
            }

            // We've reached the header of a dead term.
            //
            // If the header is for a container type, we must visit its elements
            // and drop any ref-counted pointers. If the header is for a non-container
            // type, we simply skip over the allocation and move on to the next one.
            let header = unsafe { term.as_header() };
            match header.tag() {
                Tag::Tuple => {
                    let tuple = unsafe { &*<Tuple as Boxable>::from_raw_parts(ptr.cast(), header) };
                    for element in tuple.as_slice() {
                        element.maybe_decrement_refcount();
                    }
                    iter.skip_bytes(mem::size_of_val(tuple));
                }
                Tag::Map => {
                    let map =
                        unsafe { &*<SmallMap as Boxable>::from_raw_parts(ptr.cast(), header) };
                    for element in map.keys() {
                        element.maybe_decrement_refcount();
                    }
                    for element in map.values() {
                        element.maybe_decrement_refcount();
                    }
                    iter.skip_bytes(mem::size_of_val(map));
                }
                Tag::Closure => {
                    let closure =
                        unsafe { &*<Closure as Boxable>::from_raw_parts(ptr.cast(), header) };
                    for element in closure.env() {
                        element.maybe_decrement_refcount();
                    }
                    iter.skip_bytes(mem::size_of_val(closure));
                }
                Tag::Slice => {
                    let slice =
                        unsafe { &*<BitSlice as Boxable>::from_raw_parts(ptr.cast(), header) };
                    slice.owner.maybe_decrement_refcount();
                    iter.skip_bytes(mem::size_of::<BitSlice>());
                }
                Tag::Match => {
                    let matcher =
                        unsafe { &*<MatchContext as Boxable>::from_raw_parts(ptr.cast(), header) };
                    matcher.owner.maybe_decrement_refcount();
                    iter.skip_bytes(mem::size_of::<MatchContext>());
                }
                // For the following types, skip over the allocated type
                Tag::BigInt => unsafe {
                    let ptr = <BigInt as Boxable>::from_raw_parts(ptr.cast(), header);
                    ptr::drop_in_place(ptr);
                    iter.skip_bytes(mem::size_of::<BigInt>());
                },
                Tag::Pid => unsafe {
                    let ptr = <Pid as Boxable>::from_raw_parts(ptr.cast(), header);
                    ptr::drop_in_place(ptr);
                    iter.skip_bytes(mem::size_of::<Pid>());
                },
                Tag::Port => unimplemented!(),
                Tag::Reference => unsafe {
                    let ptr: *mut Reference = ptr::from_raw_parts_mut(ptr.cast(), ());
                    ptr::drop_in_place(ptr);
                    iter.skip_bytes(mem::size_of::<Reference>());
                },
                Tag::Binary => {
                    let bin =
                        unsafe { &*<BinaryData as Boxable>::from_raw_parts(ptr.cast(), header) };
                    iter.skip_bytes(mem::size_of_val(bin));
                }
            }
        }
    }
}

pub struct HeapIter(HeapRange);
impl HeapIter {
    pub fn new(range: Range<*mut u8>) -> Self {
        Self(HeapRange {
            start: range.start.cast(),
            end: range.end.cast(),
        })
    }

    #[cfg(test)]
    pub fn from<H: Heap>(heap: &H) -> Self {
        Self(HeapRange {
            start: heap.heap_start().cast(),
            end: heap.heap_top().cast(),
        })
    }
}
impl core::iter::FusedIterator for HeapIter {}
impl Iterator for HeapIter {
    type Item = *mut OpaqueTerm;

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let diff = unsafe { self.0.end.sub_ptr(self.0.start) };
        (0, Some(diff * mem::size_of::<OpaqueTerm>()))
    }

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(ptr) = self.0.next() {
            let term = unsafe { *ptr };

            // If this is a non-literal boxed value, return it
            if term.is_box() && !term.is_literal() {
                return Some(ptr);
            }

            // If this is a header value, return it
            if term.is_header() {
                return Some(ptr);
            }

            // Otherwise, skip it
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;
    use firefly_alloc::heap::{FixedSizeHeap, Heap};

    use crate::term::*;

    use super::*;

    impl<const N: usize> Reap for FixedSizeHeap<N> {
        #[inline]
        fn reap(&self) {
            self.used_range().reap();
        }
    }

    #[test]
    fn reap_test() {
        let heap = FixedSizeHeap::<128>::default();

        let rc = BinaryData::from_str("foobar");
        let weak = Arc::downgrade(&rc);
        let bin = Term::RcBinary(rc);

        // Allocate a few terms
        let mut map = Map::with_capacity_in(4, &heap).unwrap();
        map.put_mut(Term::Int(1), Term::Bool(true));
        // Embed our ref-counted binary in a tuple
        Tuple::from_slice(
            &[atoms::Undefined.into(), Term::Int(101).into(), bin.into()],
            &heap,
        )
        .unwrap();

        {
            // We should be able to upgrade our weak right now
            assert!(weak.upgrade().is_some());
        }

        // Reap and ensure that our ref-counted object was released
        heap.reap();

        assert_eq!(weak.upgrade(), None);
    }

    #[test]
    fn heap_iter_test() {
        let heap = FixedSizeHeap::<1024>::default();

        // Allocate a variety of terms on the heap.
        //
        // The following should more or less be the heap layout in words,
        // where the topmost line is the lowest address, growing downward.
        //
        // CONS t | []
        //      h |undefined <-
        // CONS t |___________|
        //      h | 101  <----------
        // MAP    | header          |
        //        | 1               |
        //        | 2               |
        //        | 3               |
        //        | NONE            |
        //        | true            |
        //        | false           |
        //        | HeapBinary--    |
        //        | NONE        |   |
        // BINARY | header <-----   |
        //        | "abcdefgh"   |  |
        //        | "ijklmnop"   |  |
        //        | "qrstuvwx"   |  |
        //        | "yz123456"   |  |
        //        | "7890"       |  |
        // TUPLE  | header <-----|--|-
        //        | NIL          |  | |
        //        | HeapBinary <-|  | |
        //        | 5               | |
        // CONS   | ________________| |
        //        | Tuple ------------
        let mut list = ListBuilder::new(&heap);
        list.push(Term::Atom(atoms::Undefined)).unwrap();
        list.push(Term::Int(101)).unwrap();
        let mut map = Map::with_capacity_in(4, &heap).unwrap();
        map.put_mut(Term::Int(1), Term::Bool(true));
        map.put_mut(Term::Int(2), Term::Bool(false));
        let bin =
            BinaryData::from_small_str("abcdefghijklmnopqrstuvwxyz1234567890", &heap).unwrap();
        map.put_mut(Term::Int(3), Term::HeapBinary(bin));
        list.push(
            Tuple::from_slice(&[OpaqueTerm::NIL, bin.into(), Term::Int(5).into()], &heap)
                .map(Term::Tuple)
                .unwrap(),
        )
        .unwrap();
        list.finish().unwrap();

        let mut iter = HeapIter::from(&heap);

        // The first thing produced by the iterator should be the tail pointer of the second cons cell
        let term = unsafe { *iter.next().unwrap() };
        assert!(term.is_nonempty_list());
        // The next thing should be the header of the map
        let term = unsafe { *iter.next().unwrap() };
        assert!(term.is_header());
        assert_eq!(Tag::Map, unsafe { term.tag() });
        // The next will be a pointer to the heap binary
        let term = unsafe { *iter.next().unwrap() };
        assert!(term.is_gcbox());
        assert_eq!(TermType::Binary, term.r#typeof());
        // The next will be the header of the heap binary
        let term = unsafe { *iter.next().unwrap() };
        assert!(term.is_header());
        assert_eq!(Tag::Binary, unsafe { term.tag() });
        // Then the tuple header
        let term = unsafe { *iter.next().unwrap() };
        assert!(term.is_header());
        assert_eq!(Tag::Tuple, unsafe { term.tag() });
        // Then the heap binary pointer
        let term = unsafe { *iter.next().unwrap() };
        assert!(term.is_gcbox());
        assert_eq!(TermType::Binary, term.r#typeof());
        // Then the pointer to the previous tuple (head of cons cell)
        let term = unsafe { *iter.next().unwrap() };
        assert!(term.is_tuple());
        // Then the pointer to the previous list (tail of cons cell)
        let term = unsafe { *iter.next().unwrap() };
        assert!(term.is_nonempty_list());
        assert_eq!(None, iter.next());
    }
}
