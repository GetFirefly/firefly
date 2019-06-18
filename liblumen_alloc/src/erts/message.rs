#![allow(unused)]
use core::mem;

use intrusive_collections::LinkedListLink;

use super::{InvalidTermError, Term, TypedTerm};

#[derive(Debug)]
pub struct Message {
    header: usize,
    link: LinkedListLink,
    data: Term,
}
impl Message {
    const STORAGE_TYPE_SHIFT: usize = (mem::size_of::<usize>() * 8) - 1;
    const MASK_STORAGE_TYPE: usize = 1 << Self::STORAGE_TYPE_SHIFT;
    const FLAG_STORAGE_ON_HEAP: usize = 0;
    const FLAG_STORAGE_OFF_HEAP: usize = 1 << Self::STORAGE_TYPE_SHIFT;

    #[inline]
    pub fn on_heap(data: Term) -> Self {
        Self {
            header: Self::FLAG_STORAGE_ON_HEAP,
            link: LinkedListLink::new(),
            data,
        }
    }

    #[inline]
    pub fn off_heap(data: Term) -> Self {
        Self {
            header: Self::FLAG_STORAGE_OFF_HEAP,
            link: LinkedListLink::new(),
            data,
        }
    }

    #[inline]
    pub fn is_off_heap(&self) -> bool {
        self.header & Self::FLAG_STORAGE_OFF_HEAP == Self::FLAG_STORAGE_OFF_HEAP
    }

    #[inline]
    pub fn is_on_heap(&self) -> bool {
        self.header & Self::FLAG_STORAGE_ON_HEAP == Self::FLAG_STORAGE_ON_HEAP
    }

    #[inline]
    pub fn data(&self) -> Result<TypedTerm, InvalidTermError> {
        self.data.to_typed_term()
    }
}
#[cfg(debug_assertions)]
impl Drop for Message {
    fn drop(&mut self) {
        assert!(!self.link.is_linked());
    }
}
