use crate::erts::fragment;
use crate::erts::term::prelude::Term;

use intrusive_collections::UnsafeRef;

#[derive(Debug)]
pub enum Message {
    /// A message whose `data` is allocated inside the receiving process's heap.
    Process(Process),
    /// A message whose `message` `Term` had to be allocated in `heap` outside of the receiving
    /// `Process` because the `Process`'s `Heap` was locked.
    HeapFragment(HeapFragment),
}

impl Message {
    pub fn data(&self) -> &Term {
        match self {
            Self::Process(Process { data }) => data,
            Self::HeapFragment(HeapFragment { data, .. }) => data,
        }
    }
}

#[derive(Debug)]
pub struct Process {
    pub data: Term,
}

#[derive(Debug)]
pub struct HeapFragment {
    pub unsafe_ref_heap_fragment: UnsafeRef<fragment::HeapFragment>,
    pub data: Term,
}
