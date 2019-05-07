use crate::heap;
use crate::term::Term;

#[cfg_attr(test, derive(Debug))]
pub enum Message {
    /// A message whose `Term` is allocated inside the receiving `Process`'s `Heap`.
    Process(Term),
    /// A message whose `message` `Term` had to be allocated in `heap` outside of the receiving
    /// `Process` because the `Process`'s `Heap` was locked.
    Heap(Heap),
}

#[derive(Debug)]
pub struct Heap {
    pub heap: heap::Heap,
    pub message: Term,
}
