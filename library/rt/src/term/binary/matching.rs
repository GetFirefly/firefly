use alloc::alloc::{AllocError, Allocator};
use core::any::TypeId;
use core::ptr::NonNull;
use core::slice;

use liblumen_alloc::gc::GcBox;
use liblumen_binary::{Matcher, Selection};

use crate::term::{OpaqueTerm, Term};

/// This represents the structure of the result expected by generated code
/// and produced by binary matching intrinsics, it is equivalent to a multi-value
/// return with 3 values, and is used like so in generated code:
///
/// ```ignore
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

/// A slice of another binary or bitstring value
#[repr(C)]
pub struct MatchContext {
    /// This a thin pointer to the original term we're borrowing from
    /// This is necessary to properly keep the owner live, either from the perspective
    /// of the garbage collector, or reference counting, until this slice is no
    /// longer needed.
    ///
    /// If the original data is not from a term, this will be None
    owner: OpaqueTerm,
    /// We give the matcher static lifetime because we are managing the lifetime
    /// of the referenced data manually. The Rust borrow checker is of no help to
    /// us with most term data structures, due to their lifetimes being tied to a
    /// specific process heap, which can be swapped between at arbitrary points.
    /// However, our memory management strategy ensures that we never free memory
    /// that is referenced by live objects, so we are relying on that here.
    matcher: Matcher<'static>,
}
impl MatchContext {
    pub const TYPE_ID: TypeId = TypeId::of::<MatchContext>();

    pub fn new<A: Allocator>(owner: OpaqueTerm, alloc: A) -> Result<GcBox<Self>, AllocError> {
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

        GcBox::new_in(Self { owner, matcher }, alloc)
    }

    #[inline]
    pub fn owner(&self) -> OpaqueTerm {
        self.owner
    }

    #[inline]
    pub fn matcher(&mut self) -> &mut Matcher<'static> {
        &mut self.matcher
    }

    /// Returns the number of bits remaining in the underlying binary data from the current position
    #[inline]
    pub fn bits_remaining(&self) -> usize {
        self.matcher.bit_size()
    }
}
