mod builder;
mod charlist;
mod clone;
pub mod iter;

pub use self::builder::ListBuilder;
pub use self::charlist::CharlistToBinaryError;

use alloc::alloc::{AllocError, Allocator};
use core::fmt;
use core::hash::{Hash, Hasher};
use core::mem::MaybeUninit;
use core::ops::Deref;

use firefly_alloc::clone::WriteCloneIntoRaw;
use firefly_alloc::heap::Heap;

use crate::cmp::ExactEq;
use crate::gc::Gc;

use super::{OpaqueTerm, Term, TupleIndex};

/// Represents the tail of an improper list
///
/// An improper list is a list which ends with any term other than `Nil`.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImproperList {
    pub tail: Term,
}
impl fmt::Debug for ImproperList {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.tail, f)
    }
}
impl fmt::Display for ImproperList {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.tail, f)
    }
}

/// Represents a single element/cell in a linked list of cons cells.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Cons {
    pub head: OpaqueTerm,
    pub tail: OpaqueTerm,
}
impl Cons {
    /// Allocates a new cons cell in the given allocator
    pub fn new_in<A: ?Sized + Allocator>(value: Self, alloc: &A) -> Result<Gc<Self>, AllocError> {
        let mut this = Self::new_uninit_in(alloc)?;
        this.write(value);
        Ok(unsafe { this.assume_init() })
    }

    /// Allocates a new uninitialized cons cell in the given allocator
    ///
    /// NOTE: The returned cell is wrapped in `MaybeUninit` because the head/tail require
    /// initialization.
    #[inline]
    pub fn new_uninit_in<A: ?Sized + Allocator>(
        alloc: &A,
    ) -> Result<Gc<MaybeUninit<Self>>, AllocError> {
        Gc::<Self>::new_uninit_in(alloc)
    }

    /// Clones this cons cell as a new cell in the given allocator.
    pub fn clone_in<A: ?Sized + Allocator>(&self, alloc: &A) -> Result<Gc<Self>, AllocError> {
        let mut this = Self::new_uninit_in(alloc)?;
        unsafe {
            self.write_clone_into_raw(this.as_mut_ptr());
            Ok(this.assume_init())
        }
    }

    /// Constructs a list from the given slice, the output of which will be in the same order as the slice.
    pub fn from_slice<H>(slice: &[OpaqueTerm], heap: &H) -> Result<Option<Gc<Self>>, AllocError>
    where
        H: ?Sized + Heap,
    {
        let mut builder = ListBuilder::new(&heap);
        for value in slice.iter().rev().copied() {
            builder.push(value.into())?;
        }
        Ok(builder.finish())
    }
    /// During garbage collection, when a list cell is moved to the new heap, a
    /// move marker is left in the original location. For a cons cell, the move
    /// marker sets the first word to None, and the second word to a pointer to
    /// the new location.
    #[inline]
    pub fn is_move_marker(&self) -> bool {
        self.head.is_none()
    }

    /// During garbage collection, if the this cons cell is a move marker, this
    /// function provides us with the forwarding address of the list.
    ///
    /// # SAFETY
    ///
    /// The caller must guarantee that this cons cell is a move marker, or undefined
    /// behavior will result.
    #[inline]
    pub unsafe fn forwarded_to(&self) -> Gc<Self> {
        debug_assert!(self.is_move_marker());
        Gc::from_raw_parts(self.tail.as_ptr(), ())
    }

    /// Returns the head of this list as a Term
    pub fn head(&self) -> Term {
        self.head.into()
    }

    /// Returns the tail of this list as a Term
    ///
    /// NOTE: If the tail of this cell is _not_ Nil or Cons, it represents an improper list
    pub fn tail(&self) -> Term {
        self.tail.into()
    }

    /// Constructs a new cons cell with the given head/tail values
    #[inline]
    pub fn cons(head: Term, tail: Term) -> Self {
        Self {
            head: head.into(),
            tail: tail.into(),
        }
    }

    /// Traverse the list, producing a `Result<Term, ImproperList>` for each element.
    ///
    /// If the list is proper, all elements will be `Ok(Term)`, but if the list is improper,
    /// the last element produced will be `Err(ImproperList)`. This can be unwrapped to get at
    /// the contained value, or treated as an error, depending on the context.
    #[inline]
    pub fn iter(&self) -> iter::Iter<'_> {
        iter::Iter::new(self)
    }

    /// Traverse the list, producing a `Result<OpaqueTerm, OpaqueTerm>` for each element.
    ///
    /// If the list is proper, all elements will be `Ok`, otherwise, the last element produced
    /// will be `Err`.
    #[inline]
    pub fn iter_raw(&self) -> iter::RawIter<'_> {
        iter::RawIter::new(self)
    }

    #[inline]
    pub fn iter_mut<'a, 'b: 'a>(&'b mut self) -> iter::CellsIter<'a> {
        iter::CellsIter::new(self)
    }

    /// Returns true if this cell is the head of a proper list.
    ///
    /// NOTE: The cost of this function is linear in the length of the list (i.e. `O(N)`)
    pub fn is_proper(&self) -> bool {
        self.iter().all(|result| result.is_ok())
    }

    /// Calculates the length of this list, and whether it is a proper list.
    ///
    /// Returns `Ok` with the length of the list, or `Err` if improper.
    pub fn length(&self) -> Result<usize, usize> {
        let mut current = Some(self.tail);
        let mut length = 0;
        while let Some(term) = current.take() {
            length += 1;
            // End of list
            if term.is_nil() {
                break;
            }
            // Improper list
            if !term.is_nonempty_list() {
                return Err(length + 1);
            }
            let cons = unsafe { &*(term.as_ptr() as *const Self) };
            current = Some(cons.tail);
        }
        Ok(length)
    }

    /// Searches this keyword list for the first element which has a matching key
    /// at the given index.
    ///
    /// If no key is found, returns 'badarg'
    pub fn keyfind<I, K: Into<Term>>(&self, index: I, key: K) -> Result<Option<Term>, ImproperList>
    where
        I: TupleIndex + Copy,
    {
        let key = key.into();
        for result in self.iter() {
            let Term::Tuple(tuple) = result? else { continue; };
            let Some(candidate) = tuple.get_element(index).map(Into::<Term>::into) else { continue; };
            if candidate == key {
                return Ok(Some(Term::Tuple(tuple)));
            }
        }

        Ok(None)
    }
}
impl Eq for Cons {}
impl PartialEq for Cons {
    fn eq(&self, other: &Self) -> bool {
        self.head().eq(&other.head()) && self.tail().eq(&other.tail())
    }
}
impl PartialEq<Gc<Cons>> for Cons {
    fn eq(&self, other: &Gc<Cons>) -> bool {
        self.eq(other.deref())
    }
}
impl ExactEq for Cons {
    fn exact_eq(&self, other: &Self) -> bool {
        self.head().exact_eq(&other.head()) && self.tail().exact_eq(&other.tail())
    }
}
impl PartialOrd for Cons {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cons {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.iter().cmp(other.iter())
    }
}
impl Hash for Cons {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for item in self.iter() {
            item.hash(state);
        }
    }
}
impl fmt::Debug for Cons {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use core::fmt::Write;

        f.write_char('[')?;
        for (i, value) in self.iter().enumerate() {
            match value {
                Ok(value) if i > 0 => write!(f, ", {:?}", value)?,
                Ok(value) => write!(f, "{:?}", value)?,
                Err(improper) if i > 0 => write!(f, " | {:?}", improper)?,
                Err(improper) => write!(f, "{:?}", improper)?,
            }
        }
        f.write_char(']')
    }
}
impl fmt::Display for Cons {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use core::fmt::Write;

        // See https://github.com/erlang/otp/blob/b8e11b6abe73b5f6306e8833511fcffdb9d252b5/erts/emulator/beam/erl_printf_term.c#L423-443
        if self.is_printable_string() {
            f.write_char('\"')?;

            for result in self.iter() {
                // `is_printable_string` guarantees all Ok
                let element = result.unwrap();
                match element.try_into().unwrap() {
                    '\n' => f.write_str("\\\n")?,
                    '\"' => f.write_str("\\\"")?,
                    c => f.write_char(c)?,
                }
            }

            f.write_char('\"')
        } else {
            f.write_char('[')?;

            for (i, value) in self.iter().enumerate() {
                match value {
                    Ok(value) if i > 0 => write!(f, ", {}", value)?,
                    Ok(value) => write!(f, "{}", value)?,
                    Err(improper) if i > 0 => write!(f, " | {}", improper)?,
                    Err(improper) => write!(f, "{}", improper)?,
                }
            }

            f.write_char(']')
        }
    }
}
