use core::cmp;

use super::{AsTerm, Term};

#[derive(Debug, PartialEq)]
pub struct Closure {
    header: Term,
    entry: *mut (), // pointer to function entry
    next: *mut u8,  // off heap header
    arity: usize,
    env_len: usize, // the number of free variables
    creator: Term,  // pid of creator process, possible to be either Pid or ExternalPid
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
