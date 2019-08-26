pub(in crate::erts) mod atom;
pub mod binary;
mod boxed;
pub mod arch;
mod closure;
mod float;
pub mod index;
mod integer;
pub mod list;
mod map;
pub mod pid;
mod port;
pub mod reference;
pub mod resource;
mod term;
pub(in crate::erts) mod tuple;
mod typed_term;

pub use atom::*;
pub use binary::*;
pub use boxed::*;
pub use closure::*;
pub use float::*;
pub use integer::*;
pub use list::*;
pub use map::*;
pub use pid::{ExternalPid, Pid};
pub use port::*;
pub use reference::*;
pub use term::*;
pub use tuple::*;
pub use typed_term::*;

use core::alloc::Layout;
use core::fmt;
use core::mem;
use core::str::Utf8Error;

use crate::erts::exception::system::Alloc;
use crate::erts::process::HeapAlloc;

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

/// Constructs an atom.
///
/// Panics at runtime if atom cannot be allocated.
pub fn atom_unchecked(s: &str) -> Term {
    let atom = Atom::try_from_str(s).unwrap();

    unsafe { atom.as_term() }
}

pub enum BytesFromBinaryError {
    Alloc(Alloc),
    NotABinary,
    Type,
}

/// Creates a `Pid` with the given `number` and `serial`.
pub fn make_pid(number: usize, serial: usize) -> Result<Term, pid::OutOfRange> {
    Pid::new(number, serial).map(|pid| unsafe { pid.as_term() })
}

pub enum StrFromBinaryError {
    Alloc(Alloc),
    NotABinary,
    Type,
    Utf8Error(Utf8Error),
}

impl From<BytesFromBinaryError> for StrFromBinaryError {
    fn from(bytes_from_binary_error: BytesFromBinaryError) -> StrFromBinaryError {
        use BytesFromBinaryError::*;

        match bytes_from_binary_error {
            Alloc(alloc_err) => StrFromBinaryError::Alloc(alloc_err),
            NotABinary => StrFromBinaryError::NotABinary,
            Type => StrFromBinaryError::NotABinary,
        }
    }
}

/// Creates the next (local) `Pid`
pub fn next_pid() -> Term {
    let pid = pid::next();

    unsafe { pid.as_term() }
}

/// This function determines if the inner term of a boxed term contains a move marker
///
/// The value `term` should be the result of unboxing a boxed term:
///
/// ```rust,ignore
/// // The inner value is just a forwarding pointer
/// let inner = Term::make_boxed(forward_ptr);
/// // The box is then a pointer to the inner value
/// let boxed = Term::make_boxed(&inner);
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
    } else if term.is_non_empty_list() {
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
    use liblumen_core::alloc::alloc_utils::round_up_to_multiple_of;

    round_up_to_multiple_of(bytes, mem::size_of::<usize>()) / mem::size_of::<usize>()
}

pub(crate) fn layout_from_word_size(word_size: usize) -> Layout {
    let byte_size = word_size * mem::size_of::<Term>();

    unsafe { Layout::from_size_align_unchecked(byte_size, mem::align_of::<Term>()) }
}

/// Returns the size in words required to hold the non-header fields of type `T`
#[inline]
pub(crate) fn arity_of<T: Sized>() -> usize {
    to_word_size(mem::size_of::<T>() - mem::size_of::<Term>())
}

#[allow(unused)]
#[inline]
pub(crate) fn to_arch64_word_size(bytes: usize) -> usize {
    use liblumen_core::alloc::alloc_utils::round_up_to_multiple_of;

    round_up_to_multiple_of(bytes, 8) / 8
}

#[allow(unused)]
#[inline]
pub(crate) fn to_arch32_word_size(bytes: usize) -> usize {
    use liblumen_core::alloc::alloc_utils::round_up_to_multiple_of;

    round_up_to_multiple_of(bytes, 4) / 4
}
