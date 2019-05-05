#[cfg(test)]
use std::fmt::{self, Debug};
#[cfg(test)]
use std::slice::Iter;
#[cfg(test)]
use std::sync::MutexGuard;

#[cfg(test)]
use crate::heap::{CloneIntoHeap, Heap};
use crate::message::Message;
#[cfg(test)]
use crate::term::Term;

pub struct Mailbox {
    messages: Vec<Message>,
}

impl Mailbox {
    pub fn push(&mut self, message: Message) {
        self.messages.push(message)
    }

    #[cfg(test)]
    pub fn iter(&self) -> Iter<Message> {
        self.messages.iter()
    }

    #[cfg(test)]
    pub fn receive(&mut self, unlocked_heap: MutexGuard<Heap>) -> Option<Term> {
        if self.messages.is_empty() {
            None
        } else {
            let singleton: Vec<Message> = self.messages.drain(0..1).collect();

            let received = match singleton[0] {
                Message::Heap {
                    message: message_heap_message,
                    ..
                } => message_heap_message.clone_into_heap(&unlocked_heap),
                Message::Process(process_message) => process_message,
            };

            Some(received)
        }
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
        }
    }
}
