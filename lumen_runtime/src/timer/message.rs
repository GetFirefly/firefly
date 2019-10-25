use core::ptr::NonNull;

use liblumen_alloc::erts::term::prelude::Term;(

#[derive(Debug)]
pub struct HeapFragment {
    pub heap_fragment: NonNull<liblumen_alloc::erts::HeapFragment>,
    pub term: Term,
}
