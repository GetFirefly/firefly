use liblumen_diagnostics::{ByteIndex, ByteSpan};

use super::source::Source;

/// An implementation of `Scanner` for general use
pub struct Scanner<S>
{
    source: S,
    current: (ByteIndex, char),
    pending: (ByteIndex, char),
    start: ByteIndex,
    end: ByteIndex
}
impl<S> Scanner<S>
where
    S: Source,
{
    pub fn new(mut source: S) -> Self {
        let span = source.span();
        let start = span.start();
        let end = span.end();
        let current = source.read().unwrap_or((ByteIndex(0), '\0'));
        let pending = source.read().unwrap_or((ByteIndex(0), '\0'));
        Scanner {
            source,
            current,
            pending,
            start,
            end,
        }
    }

    pub fn start(&self) -> ByteIndex {
        self.start
    }

    #[inline]
    pub fn advance(&mut self) {
        self.current = self.pending;
        self.pending = match self.source.read() {
            None => (self.end, '\0'),
            Some(ic) => ic
        };
    }

    #[inline]
    pub fn pop(&mut self) -> (ByteIndex, char) {
        let current = self.current;
        self.advance();
        current
    }

    #[inline]
    pub fn peek(&self) -> (ByteIndex, char) { self.pending }

    #[inline]
    pub fn peek_next(&mut self) -> (ByteIndex, char) {
        match self.source.peek() {
            None => (self.end, '\0'),
            Some((pos, c)) => (pos, c)
        }
    }

    #[inline]
    pub fn read(&self) -> (ByteIndex, char) { self.current }

    #[inline]
    pub fn slice(&self, span: ByteSpan) -> &str { self.source.slice(span) }
}
