#![allow(unused)]

use crate::erts::{Term, ProcessControlBlock};

use super::CloneToProcess;

/// This is an implementation of a clone-on-write smart pointer,
/// but designed to be allocator-aware, specifically, it uses the
/// `CloneIntoProcess` trait to clone borrowed terms when a mutable
/// reference is aquired, ensuring that the cloned term is cloned
/// onto the appropriate process heap, not the global heap.
/// 
/// If you don't need to clone into a process heap, then you should
/// use `alloc::borrow::Cow` instead.
/// 
/// NOTE: We say "smart pointer" here, but really the structure contains
/// values of type `Term`, which may be boxed pointers, but may also be
/// immediates.
#[derive(Debug)]
pub enum Cow {
    Borrowed(Term),
    Owned(Term),
}
impl Cow {
    pub fn clone_to_process(&self, process: &mut ProcessControlBlock) -> Self {
        match *self {
            Self::Borrowed(b) => Self::Owned(b.clone_to_process(process)),
            Self::Owned(o) => Self::Owned(o)
        }
    }
}