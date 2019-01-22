use std::collections::VecDeque;

use crate::lexer::{Lexed, Lexer, Source};

pub struct TokenStream<S> {
    eof: bool,
    current: Lexer<S>,
    streams: VecDeque<Lexer<S>>,
}
impl<S> TokenStream<S>
where
    S: Source,
{
    pub fn new(current: Lexer<S>) -> Self {
        TokenStream {
            eof: false,
            current,
            streams: VecDeque::new(),
        }
    }

    pub fn include(&mut self, next: Lexer<S>) {
        if self.eof {
            self.eof = false;
        }
        let previous = std::mem::replace::<Lexer<S>>(&mut self.current, next);
        self.streams.push_front(previous);
    }
}
impl<S> Iterator for TokenStream<S>
where
    S: Source,
{
    type Item = Lexed;

    fn next(&mut self) -> Option<Self::Item> {
        if self.eof == true {
            return None;
        }
        if let Some(next) = self.current.next() {
            return Some(next);
        }
        match self.streams.pop_front() {
            None => {
                self.eof = true;
                return None;
            }
            Some(next) => {
                self.current = next;
                return self.next();
            }
        }
    }
}
