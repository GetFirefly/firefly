#[cfg(test)]
use std::fmt::{self, Debug};
#[cfg(test)]
use std::sync::MutexGuard;

#[cfg(test)]
use crate::heap::{CloneIntoHeap, Heap};
#[cfg(test)]
use crate::message;
use crate::message::Message;
#[cfg(test)]
use crate::term::Term;
use std::collections::vec_deque::{Iter, VecDeque};

pub struct Mailbox {
    messages: VecDeque<Message>,
    seen: isize,
}

impl Mailbox {
    pub fn iter(&self) -> Iter<Message> {
        self.messages.iter()
    }

    pub fn mark_seen(&mut self) {
        self.seen = (self.messages.len() as isize) - 1;
    }

    pub fn pop(&mut self) -> Option<Message> {
        self.messages.pop_front()
    }

    pub fn push(&mut self, message: Message) {
        self.messages.push_back(message)
    }

    #[cfg(test)]
    pub fn receive(&mut self, unlocked_heap: MutexGuard<Heap>) -> Option<Term> {
        if self.messages.is_empty() {
            None
        } else {
            let singleton: Vec<Message> = self.messages.drain(0..1).collect();

            let received = match singleton[0] {
                Message::Heap(message::Heap { term, .. }) => term.clone_into_heap(&unlocked_heap),
                Message::Process(process_message) => process_message,
            };

            Some(received)
        }
    }

    pub fn remove(&mut self, index: usize) -> Option<Message> {
        let removed = self.messages.remove(index);

        if let Some(_) = removed {
            if (index as isize) <= self.seen {
                self.seen -= 1;
            }
        }

        removed
    }

    pub fn seen(&self) -> isize {
        self.seen
    }

    pub fn unmark_seen(&mut self) {
        self.seen = -1;
    }
}

#[cfg(test)]
impl Debug for Mailbox {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.messages)
    }
}

impl Default for Mailbox {
    fn default() -> Mailbox {
        Mailbox {
            messages: Default::default(),
            seen: -1,
        }
    }
}
