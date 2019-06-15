use core::cmp;
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

use crate::borrow::CloneToProcess;
use crate::erts::ProcessControlBlock;

use super::{AsTerm, Term};

/// Represents boxed terms.
///
/// A `Boxed<T>` is designed around being reified from a raw
/// pointer to a typed term in `Term::to_typed_term`, and otherwise
/// only being consumed
#[derive(Debug)]
pub struct Boxed<T> {
    term: *mut T,
    literal: bool,
    _phantom: PhantomData<T>,
}
impl<T: AsTerm> Boxed<T> {
    /// This function expects to get a pointer to a typed term, as given
    /// by unmasking a raw `*mut Term` that was flagged as a boxed term.
    ///
    /// # Safety
    ///
    /// This function is unsafe, as it is built on a raw pointer to an object
    /// whose lifetime is not guaranteed to outlive the pointer. In addition,
    /// casting a raw `Term` pointer to the wrong `T` will result in undefined
    /// behavior. Only code processing raw terms should be working with this API.
    #[inline]
    pub unsafe fn from_raw(term: *mut T) -> Self {
        Self {
            term,
            literal: false,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub unsafe fn from_raw_literal(term: *mut T) -> Self {
        Self {
            term,
            literal: true,
            _phantom: PhantomData,
        }
    }

    /// Unboxes the inner pointer, returning a `NonNull<T>`
    #[inline]
    pub fn unbox(self) -> NonNull<T> {
        unsafe { NonNull::new_unchecked(self.term) }
    }
}
unsafe impl<T: AsTerm> AsTerm for Boxed<T> {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        if self.literal {
            Term::from_raw(self.term as usize | Term::FLAG_BOXED | Term::FLAG_LITERAL)
        } else {
            Term::from_raw(self.term as usize | Term::FLAG_BOXED)
        }
    }
}
impl<T: AsTerm> AsRef<T> for Boxed<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        unsafe { &*(self.term) }
    }
}
impl<T: AsTerm> AsMut<T> for Boxed<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut T {
        unsafe { &mut *(self.term) }
    }
}
impl<T: AsTerm> Deref for Boxed<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.as_ref()
    }
}
impl<T: AsTerm> DerefMut for Boxed<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.as_mut()
    }
}
impl<T: PartialEq> PartialEq for Boxed<T> {
    fn eq(&self, other: &Self) -> bool {
        let lhs = unsafe { &*self.term };
        let rhs = unsafe { &*other.term };
        lhs.eq(rhs)
    }
}
impl<T: PartialEq> PartialEq<T> for Boxed<T> {
    fn eq(&self, other: &T) -> bool {
        let lhs = unsafe { &*self.term };
        lhs.eq(other)
    }
}
impl<T: PartialOrd> PartialOrd for Boxed<T> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        let lhs = unsafe { &*self.term };
        let rhs = unsafe { &*other.term };
        lhs.partial_cmp(rhs)
    }
}
impl<T: PartialOrd> PartialOrd<T> for Boxed<T> {
    fn partial_cmp(&self, other: &T) -> Option<cmp::Ordering> {
        let lhs = unsafe { &*self.term };
        lhs.partial_cmp(other)
    }
}

impl<T: CloneToProcess> CloneToProcess for Boxed<T> {
    fn clone_to_process(&self, process: &mut ProcessControlBlock) -> Term {
        let term = unsafe { &*self.term };
        term.clone_to_process(process)
    }
}
