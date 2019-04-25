#[cfg(test)]
use std::slice::Iter;

use crate::message::Message;

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
}

impl Default for Mailbox {
    fn default() -> Mailbox {
        Mailbox {
            messages: Default::default(),
        }
    }
}
