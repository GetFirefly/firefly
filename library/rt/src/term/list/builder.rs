use alloc::alloc::AllocError;

use firefly_alloc::heap::Heap;

use crate::gc::Gc;
use crate::term::{OpaqueTerm, Term};

use super::*;

pub struct ListBuilder<'a, H: ?Sized> {
    heap: &'a H,
    tail: Option<Gc<Cons>>,
    improper: Option<OpaqueTerm>,
}
impl<'a, H: ?Sized + Heap> ListBuilder<'a, H> {
    pub fn new(heap: &'a H) -> Self {
        Self {
            heap,
            tail: None,
            improper: None,
        }
    }

    pub fn prepend(tail: Gc<Cons>, heap: &'a H) -> Self {
        Self {
            heap,
            tail: Some(tail),
            improper: None,
        }
    }

    pub fn new_improper(improper: OpaqueTerm, heap: &'a H) -> Self {
        Self {
            heap,
            tail: None,
            improper: Some(improper),
        }
    }

    pub fn push(&mut self, value: Term) -> Result<(), AllocError> {
        let value = value.move_to_heap(self.heap)?;
        unsafe { self.push_unsafe(value) }
    }

    pub unsafe fn push_unsafe<T: Into<OpaqueTerm>>(&mut self, value: T) -> Result<(), AllocError> {
        match self.tail.take() {
            None => {
                let tail = match self.improper.take() {
                    None => OpaqueTerm::NIL,
                    Some(t) => t,
                };
                // This is the first value pushed, so we need to allocate a new cell
                let mut cell = Gc::<Cons>::new_uninit_in(self.heap)?;
                cell.write(Cons {
                    head: value.into(),
                    tail,
                });
                self.tail = Some(unsafe { cell.assume_init() });
                Ok(())
            }
            Some(tail) => {
                // We're consing a new element to an existing cell
                let mut cell = Gc::<Cons>::new_uninit_in(self.heap)?;
                cell.write(Cons {
                    head: value.into(),
                    tail: tail.into(),
                });
                self.tail = Some(unsafe { cell.assume_init() });
                Ok(())
            }
        }
    }

    pub fn finish(mut self) -> Option<Gc<Cons>> {
        self.tail.take()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use firefly_alloc::heap::FixedSizeHeap;

    #[test]
    fn list_builder_builds_proper_lists() {
        let heap = FixedSizeHeap::<128>::default();
        let mut builder = ListBuilder::new(&heap);
        builder.push(Term::Int(3)).unwrap();
        builder.push(Term::Int(2)).unwrap();
        builder.push(Term::Int(1)).unwrap();
        builder.push(Term::Int(0)).unwrap();
        let list = builder.finish().unwrap();

        let mut iter = list.iter();
        assert_eq!(iter.next(), Some(Ok(Term::Int(0))));
        assert_eq!(iter.next(), Some(Ok(Term::Int(1))));
        assert_eq!(iter.next(), Some(Ok(Term::Int(2))));
        assert_eq!(iter.next(), Some(Ok(Term::Int(3))));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }
}
