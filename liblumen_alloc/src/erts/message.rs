use core::alloc::Layout;
use core::fmt::{self, Debug};
use core::ptr::{self, NonNull};

use intrusive_collections::{intrusive_adapter, LinkedListLink, UnsafeRef};

use crate::erts::exception::system::Alloc;
use crate::erts::term::Term;
use crate::mem::bit_size_of;
use crate::std_alloc;

// This adapter is used to track a queue of messages, attach to a process's mailbox.
intrusive_adapter!(pub Adapter = UnsafeRef<Message>: Message { link: LinkedListLink });

pub struct Message {
    header: usize,
    link: LinkedListLink,
    data: Term,
}
impl Message {
    const STORAGE_TYPE_SHIFT: usize = (bit_size_of::<usize>()) - 1;
    #[allow(dead_code)]
    const MASK_STORAGE_TYPE: usize = 1 << Self::STORAGE_TYPE_SHIFT;
    const FLAG_STORAGE_ON_HEAP: usize = 0;
    const FLAG_STORAGE_OFF_HEAP: usize = 1 << Self::STORAGE_TYPE_SHIFT;

    #[inline]
    pub fn on_heap(data: Term) -> Self {
        Self {
            header: Self::FLAG_STORAGE_ON_HEAP,
            link: Default::default(),
            data,
        }
    }

    #[inline]
    pub fn off_heap(data: Term) -> Self {
        Self {
            header: Self::FLAG_STORAGE_OFF_HEAP,
            link: Default::default(),
            data,
        }
    }

    pub unsafe fn alloc(self) -> Result<NonNull<Self>, Alloc> {
        let layout = Layout::new::<Self>();
        let ptr = std_alloc::alloc(layout)?.as_ptr() as *mut Self;
        ptr::write(ptr, self);

        Ok(NonNull::new_unchecked(ptr))
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
    pub fn data(&self) -> Term {
        self.data
    }
}

impl Debug for Message {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let formatted_header = if self.is_off_heap() {
            "Self::FLAG_STORAGE_OFF_HEAP"
        } else {
            "Self::FLAG_STORAGE_ON_HEAP"
        };

        f.debug_struct("Message")
            .field("header", &format_args!("{}", formatted_header))
            .field("data", &self.data)
            .finish()
    }
}

#[cfg(debug_assertions)]
impl Drop for Message {
    fn drop(&mut self) {
        assert!(!self.link.is_linked());
    }
}
