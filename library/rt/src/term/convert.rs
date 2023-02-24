use alloc::alloc::AllocError;
use alloc::sync::Arc;
use alloc::vec::Vec;

use firefly_alloc::clone::WriteCloneIntoRaw;
use firefly_alloc::heap::Heap;
use firefly_binary::Bitstring;

use smallvec::SmallVec;

use crate::gc::Gc;

use super::*;

/// This trait represents values which can be converted to `Term`.
///
/// Implementors of this trait may use `heap` to allocate if necessary.
///
/// NOTE: Implementations for heap-allocated values _must_ ensure that
/// the resulting `Term` references data on the given heap, not whatever
/// heap the value was originally allocated on.
pub trait ToTerm: Value {
    fn to_term(self, heap: &dyn Heap) -> Result<Term, AllocError>;
}

/// Any trivially convertible types implement this trait automatically
impl<T: Value + Into<Term>> ToTerm for T {
    #[inline]
    default fn to_term(self, _: &dyn Heap) -> Result<Term, AllocError> {
        Ok(self.into())
    }
}
impl ToTerm for i64 {
    #[inline]
    fn to_term(self, heap: &dyn Heap) -> Result<Term, AllocError> {
        if self > Int::MAX_SMALL || self < Int::MIN_SMALL {
            Gc::<BigInt>::new_in(self.into(), heap).map(Term::BigInt)
        } else {
            Ok(Term::Int(self))
        }
    }
}
impl ToTerm for BigInt {
    #[inline]
    fn to_term(self, heap: &dyn Heap) -> Result<Term, AllocError> {
        Gc::new_in(self, heap).map(Term::BigInt)
    }
}
impl ToTerm for Gc<BigInt> {
    #[inline]
    fn to_term(self, heap: &dyn Heap) -> Result<Term, AllocError> {
        self.clone_to_heap(heap).map(Term::BigInt)
    }
}
impl ToTerm for Pid {
    #[inline]
    fn to_term(self, heap: &dyn Heap) -> Result<Term, AllocError> {
        Gc::new_in(self, heap).map(Term::Pid)
    }
}
impl ToTerm for Gc<Pid> {
    #[inline]
    fn to_term(self, heap: &dyn Heap) -> Result<Term, AllocError> {
        self.clone_to_heap(heap).map(Term::Pid)
    }
}
impl ToTerm for Arc<Port> {
    #[inline]
    fn to_term(self, _: &dyn Heap) -> Result<Term, AllocError> {
        Ok(Term::Port(self))
    }
}
impl ToTerm for Arc<BinaryData> {
    #[inline]
    fn to_term(self, _: &dyn Heap) -> Result<Term, AllocError> {
        Ok(Term::RcBinary(self))
    }
}
impl ToTerm for Gc<BinaryData> {
    #[inline]
    fn to_term(self, heap: &dyn Heap) -> Result<Term, AllocError> {
        self.clone_to_heap(heap).map(Term::HeapBinary)
    }
}
impl ToTerm for Reference {
    #[inline]
    fn to_term(self, heap: &dyn Heap) -> Result<Term, AllocError> {
        Gc::new_in(self, heap).map(Term::Reference)
    }
}
impl ToTerm for Gc<Reference> {
    #[inline]
    fn to_term(self, heap: &dyn Heap) -> Result<Term, AllocError> {
        self.clone_to_heap(heap).map(Term::Reference)
    }
}
impl ToTerm for &Map {
    #[inline]
    fn to_term(self, heap: &dyn Heap) -> Result<Term, AllocError> {
        self.clone_to_heap(heap).map(Term::Map)
    }
}
impl ToTerm for Gc<Map> {
    #[inline]
    fn to_term(self, heap: &dyn Heap) -> Result<Term, AllocError> {
        self.clone_to_heap(heap).map(Term::Map)
    }
}
impl ToTerm for BitSlice {
    #[inline]
    fn to_term(self, heap: &dyn Heap) -> Result<Term, AllocError> {
        if self.is_owner_refcounted() {
            Ok(Term::RefBinary(Gc::new_in(self, heap)?))
        } else {
            if heap.contains(unsafe { self.owner_ptr() }) {
                Ok(Term::RefBinary(Gc::new_in(self, heap)?))
            } else {
                let selection = self.as_selection();
                let mut bin = BinaryData::with_capacity_small(selection.byte_size(), heap)?;
                bin.copy_from_selection(selection);
                Ok(Term::HeapBinary(bin))
            }
        }
    }
}
impl ToTerm for Gc<BitSlice> {
    #[inline]
    fn to_term(self, heap: &dyn Heap) -> Result<Term, AllocError> {
        if heap.contains(Gc::as_ptr(&self)) {
            Ok(Term::RefBinary(self))
        } else {
            if heap.contains(unsafe { self.owner_ptr() }) {
                let mut cloned = Gc::new_uninit_in(heap)?;
                unsafe {
                    self.deref().write_clone_into_raw(cloned.as_mut_ptr());
                    Ok(Term::RefBinary(cloned.assume_init()))
                }
            } else {
                let selection = self.as_selection();
                let mut bin = BinaryData::with_capacity_small(selection.byte_size(), heap)?;
                bin.copy_from_selection(selection);
                Ok(Term::HeapBinary(bin))
            }
        }
    }
}
impl<'a> ToTerm for &'a str {
    #[inline]
    fn to_term(self, heap: &dyn Heap) -> Result<Term, AllocError> {
        if self.as_bytes().len() > BinaryData::MAX_HEAP_BYTES {
            Ok(Term::RcBinary(BinaryData::from_str(self)))
        } else {
            BinaryData::from_small_str(self, heap).map(Term::HeapBinary)
        }
    }
}
impl<'a> ToTerm for &'a [u8] {
    #[inline]
    fn to_term(self, heap: &dyn Heap) -> Result<Term, AllocError> {
        if self.len() > BinaryData::MAX_HEAP_BYTES {
            Ok(Term::RcBinary(BinaryData::from_bytes(self)))
        } else {
            BinaryData::from_small_bytes(self, heap).map(Term::HeapBinary)
        }
    }
}
impl ToTerm for Gc<Cons> {
    #[inline]
    fn to_term(self, heap: &dyn Heap) -> Result<Term, AllocError> {
        self.clone_to_heap(heap).map(Term::Cons)
    }
}
impl ToTerm for Gc<Tuple> {
    #[inline]
    fn to_term(self, heap: &dyn Heap) -> Result<Term, AllocError> {
        self.clone_to_heap(heap).map(Term::Tuple)
    }
}
impl<T: ToTerm, U: ToTerm> ToTerm for (T, U) {
    #[inline]
    fn to_term(self, heap: &dyn Heap) -> Result<Term, AllocError> {
        let element0 = self.0.to_term(heap)?;
        let element1 = self.1.to_term(heap)?;
        Tuple::from_slice(&[element0.into(), element1.into()], heap).map(Term::Tuple)
    }
}
impl<T: ToTerm, U: ToTerm, V: ToTerm> ToTerm for (T, U, V) {
    #[inline]
    fn to_term(self, heap: &dyn Heap) -> Result<Term, AllocError> {
        let element0 = self.0.to_term(heap)?;
        let element1 = self.1.to_term(heap)?;
        let element2 = self.2.to_term(heap)?;
        Tuple::from_slice(&[element0.into(), element1.into(), element2.into()], heap)
            .map(Term::Tuple)
    }
}
impl<'a, T: ToTerm + Clone> ToTerm for &'a [T] {
    fn to_term(self, heap: &dyn Heap) -> Result<Term, AllocError> {
        let mut list = ListBuilder::new(heap);
        for element in self.iter().cloned().rev() {
            list.push(element.to_term(heap)?)?;
        }
        Ok(list.finish().map(Term::Cons).unwrap_or(Term::Nil))
    }
}
impl<T: ToTerm> ToTerm for Vec<T> {
    fn to_term(mut self, heap: &dyn Heap) -> Result<Term, AllocError> {
        let mut list = ListBuilder::new(heap);
        for element in self.drain(..).rev() {
            list.push(element.to_term(heap)?)?;
        }
        Ok(list.finish().map(Term::Cons).unwrap_or(Term::Nil))
    }
}
impl<const N: usize, T: ToTerm> ToTerm for SmallVec<[T; N]> {
    fn to_term(mut self, heap: &dyn Heap) -> Result<Term, AllocError> {
        let mut list = ListBuilder::new(heap);
        for element in self.drain(..).rev() {
            list.push(element.to_term(heap)?)?;
        }
        Ok(list.finish().map(Term::Cons).unwrap_or(Term::Nil))
    }
}
