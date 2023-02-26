use alloc::alloc::{AllocError, Allocator, Layout};
use core::fmt;
use core::mem;
use core::ptr::NonNull;
use core::slice;

use firefly_alloc::clone::WriteCloneIntoRaw;
use firefly_alloc::heap::Heap;
use firefly_binary::{Bitstring, Matcher, Selection};

use crate::gc::Gc;
use crate::term::{BinaryData, Boxable, Header, LayoutBuilder, OpaqueTerm, Tag, Term};

/// This represents the structure of the result expected by generated code
/// and produced by binary matching intrinsics, it is equivalent to a multi-value
/// return with 3 values, and is used like so in generated code:
///
/// ```text,ignore
/// let (is_err, term_or_err, match_ctx) = bs_match(...);
/// if is_err {
///   let error = cast term_or_err as *mut Exception
///   raise error
/// } else {
///   let value = cast term_or_err as <expected match type>;
/// }
/// ... proceed to another match using the updated match_ctx
/// ```
#[repr(u32)]
pub enum MatchResult {
    Ok {
        extracted: OpaqueTerm,
        context: NonNull<MatchContext>,
    } = 0,
    Err {
        none: OpaqueTerm,
        context: NonNull<MatchContext>,
    } = 1,
}
impl MatchResult {
    #[inline(always)]
    pub fn ok(extracted: OpaqueTerm, context: NonNull<MatchContext>) -> Self {
        Self::Ok { extracted, context }
    }

    #[inline(always)]
    pub fn err(context: NonNull<MatchContext>) -> Self {
        Self::Err {
            none: OpaqueTerm::NONE,
            context,
        }
    }
}
impl fmt::Debug for MatchResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Ok { extracted, .. } => {
                let t: Term = (*extracted).into();
                write!(f, "Ok({:?})", &t)
            }
            Self::Err { .. } => {
                write!(f, "Err")
            }
        }
    }
}

/// A slice of another binary or bitstring value
#[repr(C)]
#[derive(Clone)]
pub struct MatchContext {
    pub(crate) header: Header,
    /// This a thin pointer to the original term we're borrowing from
    /// This is necessary to properly keep the owner live, either from the perspective
    /// of the garbage collector, or reference counting, until this slice is no
    /// longer needed.
    ///
    /// If the original data is not from a term, this will be None
    pub(crate) owner: OpaqueTerm,
    /// We give the matcher static lifetime because we are managing the lifetime
    /// of the referenced data manually. The Rust borrow checker is of no help to
    /// us with most term data structures, due to their lifetimes being tied to a
    /// specific process heap, which can be swapped between at arbitrary points.
    /// However, our memory management strategy ensures that we never free memory
    /// that is referenced by live objects, so we are relying on that here.
    pub(crate) matcher: Matcher<'static>,
}
impl fmt::Debug for MatchContext {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("MatchContext")
            .field("header", &self.header)
            .field("owner", &format_args!("{}", self.owner))
            .field("matcher", &self.matcher.selection)
            .finish()
    }
}
impl MatchContext {
    pub fn new<A: ?Sized + Allocator>(
        owner: OpaqueTerm,
        alloc: &A,
    ) -> Result<Gc<Self>, AllocError> {
        let term: Term = owner.into();
        let data = term
            .as_bitstring()
            .expect("unexpected input to match context");
        let bit_size = data.bit_size();
        let matcher = if bit_size == 0 {
            let selection = Selection::Empty;
            Matcher::new(selection)
        } else {
            let bytes = unsafe { data.as_bytes_unchecked() };
            let bytes = unsafe { slice::from_raw_parts::<'static>(bytes.as_ptr(), bytes.len()) };
            let selection = Selection::new(bytes, 0, data.bit_offset(), None, bit_size).unwrap();
            Matcher::new(selection)
        };

        Gc::new_in(
            Self {
                header: Header::new(Tag::Match, 0),
                owner,
                matcher,
            },
            alloc,
        )
    }

    #[inline]
    pub fn owner(&self) -> OpaqueTerm {
        self.owner
    }

    #[inline]
    pub fn matcher(&self) -> &Matcher<'static> {
        &self.matcher
    }

    #[inline]
    pub fn matcher_mut(&mut self) -> &mut Matcher<'static> {
        &mut self.matcher
    }

    /// Returns the number of bits remaining in the underlying binary data from the current position
    #[inline]
    pub fn bits_remaining(&self) -> usize {
        self.matcher.bit_size()
    }
}
impl Boxable for MatchContext {
    type Metadata = ();

    const TAG: Tag = Tag::Match;

    fn header(&self) -> &Header {
        &self.header
    }

    fn header_mut(&mut self) -> &mut Header {
        &mut self.header
    }

    fn layout_excluding_heap<H: ?Sized + Heap>(&self, heap: &H) -> Layout {
        if heap.contains((self as *const Self).cast()) {
            return Layout::new::<()>();
        }
        let mut builder = LayoutBuilder::new();
        if self.owner.is_rc() || self.owner.is_literal() {
            builder += Layout::new::<Self>();
        } else {
            assert!(self.owner.is_gcbox());
            let ptr = unsafe { self.owner.as_ptr() };
            if heap.contains(ptr.cast_const()) {
                builder += Layout::new::<Self>();
            } else {
                let byte_size = self.matcher.byte_size();
                assert!(byte_size <= BinaryData::MAX_HEAP_BYTES);
                builder.build_heap_binary(byte_size);
                builder += Layout::new::<Self>();
            }
        }
        builder.finish()
    }

    fn unsafe_clone_to_heap<H: ?Sized + Heap>(&self, heap: &H) -> Gc<Self> {
        let ptr = self as *const Self;
        if heap.contains(ptr.cast()) {
            return unsafe { Gc::from_raw(ptr.cast_mut()) };
        }

        if self.owner.is_rc() || self.owner.is_literal() {
            let mut cloned = Gc::new_uninit_in(heap).unwrap();
            unsafe {
                self.owner.maybe_increment_refcount();
                self.write_clone_into_raw(cloned.as_mut_ptr());
                return cloned.assume_init();
            }
        }

        assert!(self.owner.is_gcbox());
        let is_moved;
        let owner;
        match self.owner.move_marker() {
            Some(dest) => {
                is_moved = true;
                let boxed = unsafe { Gc::from_raw(dest.as_ptr()) };
                owner = boxed.into();
            }
            None => {
                is_moved = false;
                owner = self.owner;
            }
        }

        // If the owner hasn't moved, we need to clone both the context
        // and the original binary data.
        if !is_moved {
            return recreate_on_target_heap(self.matcher.selection, heap);
        }

        let prev_ptr = unsafe { self.owner.as_ptr() };
        let prev_header = unsafe { *prev_ptr.cast::<OpaqueTerm>() };
        let prev_header = unsafe { prev_header.as_header() };

        let new_ptr = unsafe { owner.as_ptr() };
        let new_header = unsafe { *new_ptr.cast::<OpaqueTerm>() };
        assert!(new_header.is_header());
        assert!(heap.contains(new_ptr.cast_const()));
        let new_header = unsafe { new_header.as_header() };

        let prev_owner =
            unsafe { &*<BinaryData as Boxable>::from_raw_parts(prev_ptr, prev_header) };
        let new_owner = unsafe { &*<BinaryData as Boxable>::from_raw_parts(new_ptr, new_header) };
        let prev_bytes = unsafe { prev_owner.as_bytes_unchecked() };
        let new_bytes = unsafe { new_owner.as_bytes_unchecked() };
        let original_selection = self.matcher.selection;
        let new_selection = match original_selection {
            Selection::Empty => Selection::Empty,
            Selection::Byte(b) => Selection::Byte(b),
            Selection::AlignedBinary(b) => {
                let ptr = b.as_ptr();
                let byte_offset = unsafe { prev_bytes.as_ptr().sub_ptr(ptr) };
                let byte_len = b.len();
                let start = unsafe { new_bytes.as_ptr().add(byte_offset) };
                let bytes = unsafe { slice::from_raw_parts::<'static>(start, byte_len) };
                Selection::AlignedBinary(bytes)
            }
            Selection::Binary(l, b, r) => {
                let ptr = b.as_ptr();
                let byte_offset = unsafe { prev_bytes.as_ptr().sub_ptr(ptr) };
                let byte_len = b.len();
                let start = unsafe { new_bytes.as_ptr().add(byte_offset) };
                let bytes = unsafe { slice::from_raw_parts::<'static>(start, byte_len) };
                Selection::Binary(l, bytes, r)
            }
            Selection::AlignedBitstring(b, r) => {
                let ptr = b.as_ptr();
                let byte_offset = unsafe { prev_bytes.as_ptr().sub_ptr(ptr) };
                let byte_len = b.len();
                let start = unsafe { new_bytes.as_ptr().add(byte_offset) };
                let bytes = unsafe { slice::from_raw_parts::<'static>(start, byte_len) };
                Selection::AlignedBitstring(bytes, r)
            }
            Selection::Bitstring(l, b, r) => {
                let ptr = b.as_ptr();
                let byte_offset = unsafe { prev_bytes.as_ptr().sub_ptr(ptr) };
                let byte_len = b.len();
                let start = unsafe { new_bytes.as_ptr().add(byte_offset) };
                let bytes = unsafe { slice::from_raw_parts::<'static>(start, byte_len) };
                Selection::Bitstring(l, bytes, r)
            }
        };
        let mut cloned = Gc::new_uninit_in(heap).unwrap();
        unsafe {
            cloned.write(Self {
                header: Header::new(Tag::Match, 0),
                owner,
                matcher: Matcher::new(new_selection),
            });
            cloned.assume_init()
        }
    }
}

/// Clone selected region of owner binary to target heap and create a new matcher with it
fn recreate_on_target_heap<H: ?Sized + Heap>(
    selection: Selection<'_>,
    heap: &H,
) -> Gc<MatchContext> {
    let mut new = BinaryData::with_capacity_small(selection.byte_size(), heap).unwrap();
    new.copy_from_selection(selection);
    let selection = Selection::from_bitstring(&new);
    let mut cloned = Gc::new_uninit_in(heap).unwrap();
    unsafe {
        cloned.write(MatchContext {
            header: Header::new(Tag::Match, 0),
            owner: new.into(),
            matcher: Matcher::new(mem::transmute::<_, Selection<'static>>(selection)),
        });
        cloned.assume_init()
    }
}
