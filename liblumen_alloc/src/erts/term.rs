mod atom;
mod binary;
mod closure;
mod float;
mod integer;
mod list;
mod pid;
mod port;
mod reference;
mod term;
mod tuple;
mod typed_term;
//mod map;
mod boxed;

pub use atom::*;
pub use binary::*;
pub use closure::*;
pub use float::*;
pub use integer::*;
pub use list::*;
pub use pid::*;
pub use port::*;
pub use reference::*;
pub use term::*;
pub use tuple::*;
pub use typed_term::*;
//pub use map::*;
pub use boxed::*;

use core::fmt;

#[derive(Clone, Copy)]
pub struct BadArgument(Term);
impl BadArgument {
    #[inline]
    pub fn new(term: Term) -> Self {
        Self(term)
    }

    #[inline]
    pub fn argument(&self) -> Term {
        self.0
    }
}
impl fmt::Display for BadArgument {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "bad argument: {:?}", self.0)
    }
}

/// This error is returned when attempting to convert a `Term`
/// to a concrete type, or to `TypedTerm`, and indicates the
/// various failure modes possible
#[derive(Debug, Clone, Copy)]
pub enum InvalidTermError {
    InvalidTag,
}

/// A trait which represents casting a type to a `Term`,
/// which is a generic tagged pointer type for Erlang terms.
///
/// # Safety
///
/// This trait is unsafe because it is extremely bad news to
/// do this incorrectly. If the type tag is missing, you will
/// get undefined behavior; if the type tag is incorrect, or
/// tries to treat one data type as another incompatible type,
/// you will get undefined behavior. In short, do not implement
/// this trait unless you know what you are doing, and even then
/// be very careful to test the implementation fully.
pub unsafe trait AsTerm {
    /// This is the sole function in the `AsTerm` trait, it takes
    /// a reference to `self` and expects to get a `Term` which can
    /// be subsequently stored in a process heap or elsewhere for
    /// an indeterminate amount of time.
    ///
    /// # Safety
    ///
    /// It is absolutely essential that when you create a `Term`, that
    /// the memory it points to is not stored on the Rust stack or
    /// managed by Rust. The memory must be allocated via one of the
    /// ERTS allocators, or on a process heap which will own the term.
    /// If the memory referenced gets freed by Rust, you will get undefined
    /// behavior.
    unsafe fn as_term(&self) -> Term;
}

/// Placeholder for map header
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct MapHeader(usize);
unsafe impl AsTerm for MapHeader {
    unsafe fn as_term(&self) -> Term {
        Term::from_raw(self.0 | Term::FLAG_MAP)
    }
}
impl crate::borrow::CloneToProcess for MapHeader {
    fn clone_to_process(&self, _process: &mut super::ProcessControlBlock) -> Term {
        unimplemented!()
    }
}

/// This function determines if the inner term of a boxed term contains a move marker
///
/// The value `term` should be the result of unboxing a boxed term:
///
/// ```rust,ignore
/// // The inner value is just a forwarding pointer
/// let inner = Term::from_raw(forward_ptr as usize | Term::FLAG_BOXED);
/// // The box is then a pointer to the inner value
/// let boxed = Term::from_raw(&inner as *const _ as usize | Term::FLAG_BOXED);
/// // We resolved this pointer the short way, but in practice we get `*mut Term`
/// let ptr = &boxed as *const _ as *mut Term;
/// // Dereference the box
/// let boxed = *ptr;
/// // Then extract the boxed pointer (type is `*mut Term`)
/// let inner = boxed.boxed_val();
/// // This is the term on which `is_move_marker` is intended to be tested, in
/// // other scenario, the result is misleading at best
/// assert!(is_move_marker(*inner));
/// ```
#[inline]
pub(crate) fn is_move_marker(term: Term) -> bool {
    !term.is_header()
}

/// Resolve a term potentially containing a move marker to the location
/// of the forward reference, returning the "real" term there.
///
/// Move markers are used in two scenarios:
///
/// - For non-cons cell terms which are moved, the original location is
/// updated with a box that points to the new location. There is no marker
/// per se, we just treat the term as a box
/// - For cons cells, the old cell is overwritten with a special marker
/// cell, where the head term is the none value, and the tail term is a pointer
/// to the new location of the cell
///
/// This function does not follow boxes, it just returns them as if they had
/// been found that way. In the case of a cons cell, the term you get back will
/// be the top-level list term, i.e. the term which has the pointer to the head
/// cons cell
#[inline]
pub(crate) fn follow_moved(term: Term) -> Term {
    if term.is_boxed() {
        let ptr = term.boxed_val();
        let boxed = unsafe { *ptr };
        if is_move_marker(boxed) {
            boxed
        } else {
            term
        }
    } else if term.is_list() {
        let ptr = term.list_val();
        let cons = unsafe { &*ptr };
        if cons.is_move_marker() {
            cons.tail
        } else {
            term
        }
    } else {
        term
    }
}

/// Given a number of bytes `bytes`, returns the number of words
/// needed to hold that number of bytes, rounding up if necessary
#[inline]
pub(crate) fn to_word_size(bytes: usize) -> usize {
    use core::mem;
    use liblumen_core::alloc::alloc_utils::round_up_to_multiple_of;

    round_up_to_multiple_of(bytes, mem::size_of::<usize>()) / mem::size_of::<usize>()
}

/// Returns the size in words required to hold a value of type `T`
#[inline]
pub(crate) fn word_size_of<T: Sized>() -> usize {
    use core::mem;
    to_word_size(mem::size_of::<T>())
}