use core::default::Default;

use alloc::collections::vec_deque::Iter;
use alloc::collections::VecDeque;

use crate::borrow::CloneToProcess;
use crate::erts::exception::system::Alloc;
use crate::erts::message::{self, Message};
use crate::erts::process::Process;
use crate::erts::term::Term;

#[derive(Debug)]
pub struct Mailbox {
    messages: VecDeque<Message>,
    seen: isize,

    cursor: usize,
}

impl Mailbox {
    // Start receive implementation for the eir interpreter
    pub fn recv_start(&self) {
        debug_assert!(self.cursor == 0);
    }
    /// Important to remember that this might return a term in a heap
    /// fragment, and that it needs to be copied over to the process
    /// heap before the message is removed from the mailbox.
    pub fn recv_peek(&self) -> Option<Term> {
        match self.messages.get(self.cursor) {
            None => None,
            Some(Message::Process(message::Process { data })) => Some(*data),
            Some(Message::HeapFragment(message::HeapFragment { data, .. })) => Some(*data),
        }
    }
    pub fn recv_last_off_heap(&self) -> bool {
        match &self.messages[self.cursor - 1] {
            Message::Process(_) => false,
            Message::HeapFragment(_) => true,
        }
    }
    pub fn recv_increment(&mut self) {
        self.cursor += 1;
    }
    pub fn recv_finish(&mut self, proc: &Process) {
        self.remove(self.cursor - 1, proc);
        self.cursor = 0;
    }
    // End receive implementation for the eir interpreter

    pub fn flush<F>(&mut self, predicate: F, process: &Process) -> bool
    where
        F: Fn(&Message) -> bool,
    {
        match self.iter().position(predicate) {
            Some(index) => {
                self.remove(index, process);

                true
            }
            None => false,
        }
    }

    pub fn iter(&self) -> Iter<Message> {
        self.messages.iter()
    }

    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn mark_seen(&mut self) {
        self.seen = (self.len() as isize) - 1;
    }

    /// Pops the `message` out of the mailbox from the front of the queue.
    pub fn pop(&mut self) -> Option<Message> {
        match self.messages.pop_front() {
            option_message @ Some(_) => {
                self.decrement_seen();

                option_message
            }
            None => None,
        }
    }

    /// Puts `message` into mailbox at end of receive queue.
    pub fn push(&mut self, message: Message) {
        self.messages.push_back(message);
    }

    /// Pops the `message` out of the mailbox from the front of the queue AND clones it into
    /// `heap_guard` heap.
    pub fn receive(&mut self, process: &Process) -> Option<Result<Term, Alloc>> {
        self.messages.pop_front().map(|message| match message {
            Message::Process(message::Process { data }) => {
                self.decrement_seen();

                Ok(data)
            }
            Message::HeapFragment(message::HeapFragment {
                ref unsafe_ref_heap_fragment,
                data,
            }) => match data.clone_to_heap(&mut process.acquire_heap()) {
                Ok(heap_data) => {
                    let mut off_heap = process.off_heap.lock();

                    unsafe {
                        let mut cursor =
                            off_heap.cursor_mut_from_ptr(unsafe_ref_heap_fragment.as_ref());
                        cursor
                            .remove()
                            .expect("HeapFragment was not in process's off_heap");
                    }

                    self.decrement_seen();

                    Ok(heap_data)
                }
                err @ Err(_) => {
                    self.messages.push_front(message);

                    err
                }
            },
        })
    }

    pub fn remove(&mut self, index: usize, process: &Process) {
        let message = self.messages.remove(index).unwrap();

        if let Message::HeapFragment(message::HeapFragment {
            unsafe_ref_heap_fragment,
            ..
        }) = message
        {
            let mut off_heap = process.off_heap.lock();

            unsafe {
                let mut cursor = off_heap.cursor_mut_from_ptr(unsafe_ref_heap_fragment.as_ref());
                cursor
                    .remove()
                    .expect("HeapFragment was not in process's off_heap");
            }
        }

        if (index as isize) <= self.seen {
            self.seen -= 1;
        }
    }

    pub fn seen(&self) -> isize {
        self.seen
    }

    pub fn unmark_seen(&mut self) {
        self.seen = -1;
    }

    // Private

    fn decrement_seen(&mut self) {
        if 0 <= self.seen {
            self.seen -= 1;
        }
    }
}

impl Default for Mailbox {
    fn default() -> Mailbox {
        Mailbox {
            messages: Default::default(),
            seen: -1,
            cursor: 0,
        }
    }
}
