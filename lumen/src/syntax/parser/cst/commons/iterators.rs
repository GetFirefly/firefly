use crate::syntax::tokenizer::values::Symbol;

use super::parts::{ConsCell, ConsCellTail, Sequence, SequenceTail};

#[derive(Debug)]
pub struct ConsCellIter<'a, T: 'a>(ConsCellIterInner<'a, T>);
impl<'a, T: 'a> ConsCellIter<'a, T> {
    pub fn new(head: &'a ConsCell<T>) -> Self {
        let inner = ConsCellIterInner::Head(head);
        ConsCellIter(inner)
    }
}
impl<'a, T: 'a> Iterator for ConsCellIter<'a, T> {
    type Item = (Option<Symbol>, &'a T);
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}
#[derive(Debug)]
enum ConsCellIterInner<'a, T: 'a> {
    Head(&'a ConsCell<T>),
    Tail(&'a ConsCellTail<T>),
    Eos,
}
impl<'a, T: 'a> Iterator for ConsCellIterInner<'a, T> {
    type Item = (Option<Symbol>, &'a T);
    fn next(&mut self) -> Option<Self::Item> {
        match *self {
            ConsCellIterInner::Head(&ConsCell { ref item, ref tail }) => {
                if let Some(ref tail) = *tail {
                    *self = ConsCellIterInner::Tail(tail);
                } else {
                    *self = ConsCellIterInner::Eos
                }
                Some((None, item))
            }
            ConsCellIterInner::Tail(&ConsCellTail::Proper {
                ref item, ref tail, ..
            }) => {
                if let Some(ref tail) = *tail {
                    *self = ConsCellIterInner::Tail(tail);
                } else {
                    *self = ConsCellIterInner::Eos
                }
                Some((Some(Symbol::Comma), item))
            }
            ConsCellIterInner::Tail(&ConsCellTail::Improper { ref item, .. }) => {
                *self = ConsCellIterInner::Eos;
                Some((Some(Symbol::VerticalBar), item))
            }
            ConsCellIterInner::Eos => None,
        }
    }
}

#[derive(Debug)]
pub struct SequenceIter<'a, T: 'a, D: 'a>(SequenceIterInner<'a, T, D>);
impl<'a, T: 'a, D: 'a> SequenceIter<'a, T, D> {
    pub fn new(seq: &'a Sequence<T, D>) -> Self {
        let inner = SequenceIterInner::Head(seq);
        SequenceIter(inner)
    }
}
impl<'a, T: 'a, D: 'a> Iterator for SequenceIter<'a, T, D> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

#[derive(Debug)]
enum SequenceIterInner<'a, T: 'a, D: 'a> {
    Head(&'a Sequence<T, D>),
    Tail(&'a SequenceTail<T, D>),
    Eos,
}
impl<'a, T: 'a, D: 'a> Iterator for SequenceIterInner<'a, T, D> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        match *self {
            SequenceIterInner::Head(&Sequence { ref item, ref tail }) => {
                if let Some(ref tail) = *tail {
                    *self = SequenceIterInner::Tail(tail);
                } else {
                    *self = SequenceIterInner::Eos
                }
                Some(item)
            }
            SequenceIterInner::Tail(&SequenceTail {
                ref item, ref tail, ..
            }) => {
                if let Some(ref tail) = *tail {
                    *self = SequenceIterInner::Tail(tail);
                } else {
                    *self = SequenceIterInner::Eos
                }
                Some(item)
            }
            SequenceIterInner::Eos => None,
        }
    }
}
