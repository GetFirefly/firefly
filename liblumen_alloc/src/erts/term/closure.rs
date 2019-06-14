use core::cmp;
use core::ptr;
use core::mem;

use super::{AsTerm, Term};

use crate::borrow::CloneToProcess;
use crate::erts::ProcessControlBlock;

#[derive(Debug, PartialEq)]
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
        Term::from_raw((self as *const _ as usize) | Term::FLAG_CLOSURE)
    }
}
impl PartialOrd for Closure {
    fn partial_cmp(&self, _other: &Closure) -> Option<cmp::Ordering> {
        None
    }
}
impl CloneToProcess for Closure {
    fn clone_to_process(&self, process: &mut ProcessControlBlock) -> Term {
        // Allocate space on process heap
        let bytes = mem::size_of::<Self>() + (self.env_len * mem::size_of::<Term>());
        let mut words = bytes / mem::size_of::<Term>();
        if bytes % mem::size_of::<Term>() != 0 {
            words += 1;
        }
        unsafe {
            let ptr = process.alloc(words).unwrap().as_ptr();
            // Copy to newly allocated region
            ptr::copy_nonoverlapping(self as *const _ as *const u8, ptr as *mut u8, bytes);
            // Return term
            let closure = &*(ptr as *mut Self);
            closure.as_term()
        }
    }
}