use crate::erts::message::{Message, MessageAdapter, MessageData};

use intrusive_collections::linked_list::Cursor;
use intrusive_collections::{LinkedList, UnsafeRef};

use liblumen_arena::TypedArena;

pub struct Mailbox {
    len: usize,
    messages: LinkedList<MessageAdapter>,
    storage: TypedArena<Message>,
}
impl Mailbox {
    /// Create a new, empty mailbox
    #[inline]
    pub fn new() -> Self {
        Self {
            len: 0,
            messages: LinkedList::new(MessageAdapter::new()),
            storage: TypedArena::default(),
        }
    }

    /// Returns true if the mailbox is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Returns the number of messages in the mailbox
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns a cursor starting at the oldest message in the mailbox
    pub fn cursor(&self) -> Cursor<'_, MessageAdapter> {
        self.messages.back()
    }

    /// Returns a cursor starting from the node given by the provided pointer
    pub unsafe fn cursor_from_ptr(&mut self, ptr: *const Message) -> Cursor<'_, MessageAdapter> {
        self.messages.cursor_from_ptr(ptr)
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = &Message> + 'a {
        self.messages.iter()
    }

    /// Appends the given message to the mailbox queue
    pub fn push(&mut self, data: MessageData) {
        let ptr = self.storage.alloc(Message::new(data));
        self.messages
            .push_front(unsafe { UnsafeRef::from_raw(ptr) });
        self.len += 1;
    }

    /// Removes the given message from the mailbox
    pub fn remove(&mut self, message: *const Message) {
        let mut cursor = unsafe { self.messages.cursor_mut_from_ptr(message) };
        debug_assert!(!cursor.is_null());
        cursor.remove();
        self.len -= 1;
    }

    /// Removes the first matching message from the mailbox, traversing in receive order (oldest->newest)
    pub fn flush<F>(&mut self, predicate: F) -> bool
    where
        F: Fn(&Message) -> bool,
    {
        let mut current = self.messages.back_mut();
        loop {
            if current.is_null() {
                break;
            }
            let found = current.get().map(|msg| predicate(msg)).unwrap_or(false);
            if found {
                current.remove();
                self.len -= 1;
                return found;
            }
            current.move_prev();
        }

        false
    }

    /// This function garbage collects the storage arena to ensure it doesn't grow
    /// indefinitely. It uses the messages in the mailbox as roots.
    ///
    /// It is assumed that this is run _before_ the heap fragments associated with any
    /// messages in the arena are dropped, so this needs to be done before off_heap is
    /// cleared.
    pub fn garbage_collect(&mut self) {
        let mut len = 0;
        let storage = TypedArena::with_capacity(self.len);
        let mut messages = LinkedList::new(MessageAdapter::new());
        // Walk the messages list from back-to-front, cloning message references
        // into the new storage arena, and pushing them on the front of the new message
        // list. When complete, we should have all of the messages in the new list,
        // in the same order, only requiring a single traversal.
        let mut cursor = self.messages.back();
        while let Some(message) = cursor.get() {
            let ptr = storage.alloc(message.clone());
            messages.push_front(unsafe { UnsafeRef::from_raw(ptr) });
            len += 1;
            cursor.move_prev();
        }
        // We don't need to unlink/free objects in the list, so use the faster version here
        self.messages.fast_clear();
        self.messages = messages;
        // This shouldn't actually be necessary, but for sanity we recalculate len at the
        // same time we rebuild the mailbox
        self.len = len;
        // This will cause the old storage to drop, which will deallocate potentially many
        // chunks, depending on how long the arena was growing. If this happens on a busy
        // process with a lot of contenders for the mailbox lock, it could cause problems
        self.storage = storage;
    }
}
impl Default for Mailbox {
    fn default() -> Self {
        Self::new()
    }
}
