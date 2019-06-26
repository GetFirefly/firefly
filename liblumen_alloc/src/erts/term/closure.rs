use core::alloc::AllocErr;
use core::cmp;
use core::fmt;
use core::mem;
use core::ptr;

use super::{AsTerm, Term};

use crate::borrow::CloneToProcess;
use crate::erts::{to_word_size, HeapAlloc};

#[derive(PartialEq)]
pub struct Closure {
    header: Term,
    entry: *mut (), // pointer to function entry
    next: *mut u8,  // off heap header
    arity: usize,
    creator: Term,  // pid of creator process, possible to be either Pid or ExternalPid
    env_len: usize, // the number of free variables
    env: *mut Term, // pointer to first element of free variable array
}
unsafe impl AsTerm for Closure {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::make_boxed(self as *const Self)
    }
}
impl PartialOrd for Closure {
    fn partial_cmp(&self, _other: &Closure) -> Option<cmp::Ordering> {
        None
    }
}
impl CloneToProcess for Closure {
    fn clone_to_heap<A: HeapAlloc>(&self, heap: &mut A) -> Result<Term, AllocErr> {
        // Allocate space on process heap
        let words = self.size_in_words();
        let bytes = words * mem::size_of::<usize>();
        unsafe {
            let ptr = heap.alloc(words)?.as_ptr();
            // Copy to newly allocated region
            ptr::copy_nonoverlapping(self as *const _ as *const u8, ptr as *mut u8, bytes);
            // Return term
            Ok(Term::make_boxed(ptr))
        }
    }

    fn size_in_words(&self) -> usize {
        let mut size = to_word_size(mem::size_of::<Self>());
        for offset in 0..self.env_len {
            let ptr = unsafe { self.env.offset(offset as isize) };
            let term = unsafe { &*ptr };
            size += term.size_in_words()
        }
        size
    }
}
impl fmt::Debug for Closure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Closure")
            .field("header", &self.header.as_usize())
            .field("entry", &self.entry)
            .field("next", &self.next)
            .field("arity", &self.arity)
            .field("creator", &self.creator)
            .field("env_len", &self.env_len)
            .field("env", &self.env)
            .finish()
    }
}
