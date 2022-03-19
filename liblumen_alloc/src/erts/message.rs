use crate::erts::fragment;
use crate::erts::term::prelude::Term;

use intrusive_collections::intrusive_adapter;
use intrusive_collections::{LinkedListLink, UnsafeRef};

/// This struct represents a single message, potentially attached to a mailbox
///
/// NOTE: This struct is accessed from generated code, so its layout is expected
/// to remain stable. The `link` is two words, and `data` is three words. On 64-bit
/// architectures, data is equivalent to `{i32, [5 x i32]}`, the first i32 is the
/// enum discriminator, the second i32 is padding, and starting at the 3rd i32 is
/// either a single Term, or the HeapFragment struct.
#[derive(Clone)]
#[repr(C)]
pub struct Message {
    pub link: LinkedListLink,
    pub data: MessageData,
}
impl Message {
    pub fn new(data: MessageData) -> Self {
        Self {
            link: LinkedListLink::default(),
            data,
        }
    }

    pub fn data(&self) -> Term {
        match self.data {
            MessageData::Process(data) => data,
            MessageData::HeapFragment(HeapFragment { data, .. }) => data,
        }
    }

    pub fn is_off_heap(&self) -> bool {
        match self.data {
            MessageData::HeapFragment(_) => true,
            _ => false,
        }
    }
}

// This adapter is the means by which we can store messages in the mailbox via `link`
intrusive_adapter!(pub MessageAdapter = UnsafeRef<Message>: Message { link: LinkedListLink });

/// This struct represents where message data is stored, with the means to access it
#[derive(Debug, Clone)]
#[repr(C)]
pub enum MessageData {
    /// A message whose `data` is allocated inside the receiving process's heap.
    Process(Term),
    /// A message whose `message` `Term` had to be allocated in `heap` outside of the receiving
    /// `Process` because the `Process`'s `Heap` was locked.
    HeapFragment(HeapFragment),
}

/// This struct is used to represent data which is stored in a heap fragment
///
/// The boxed term will have all its data on the linked heap fragment. The
/// heap fragment will be attached to a processes' off_heap list
#[derive(Debug, Clone)]
#[repr(C)]
pub struct HeapFragment {
    pub data: Term,
    pub unsafe_ref_heap_fragment: UnsafeRef<fragment::HeapFragment>,
}
