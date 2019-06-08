use std::cmp::Ordering::{self, *};
use std::convert::{Into, TryFrom, TryInto};
#[cfg(test)]
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
use std::iter::FusedIterator;
use std::ops::Index;

use crate::exception::{self, Exception};
use crate::heap::{CloneIntoHeap, Heap};
use crate::integer::Integer;
use crate::process::Process;
use crate::term::{Tag::*, Term};

#[repr(C)]
pub struct Tuple {
    arity: Term, // the elements follow, but can't be represented in Rust
}

#[repr(transparent)]
pub struct ZeroBasedIndex(usize);

impl TryFrom<Term> for ZeroBasedIndex {
    type Error = Exception;

    fn try_from(term: Term) -> Result<ZeroBasedIndex, Exception> {
        let OneBasedIndex(one_based_index) = term.try_into()?;

        Ok(ZeroBasedIndex(one_based_index - 1))
    }
}

#[repr(transparent)]
pub struct OneBasedIndex(usize);

impl TryFrom<Term> for OneBasedIndex {
    type Error = Exception;

    fn try_from(term: Term) -> Result<OneBasedIndex, Exception> {
        let one_based_index_usize: usize = term.try_into()?;

        if 1 <= one_based_index_usize {
            Ok(OneBasedIndex(one_based_index_usize))
        } else {
            Err(badarg!())
        }
    }
}

impl Tuple {
    pub fn from_slice(element_slice: &[Term], heap: &Heap) -> &'static Tuple {
        let arity = element_slice.len();
        let arity_term = Term::arity(arity);
        let mut term_vector = Vec::with_capacity(1 + arity);

        term_vector.push(arity_term);
        term_vector.extend_from_slice(element_slice);

        let pointer = heap.alloc_term_slice(term_vector.as_slice()) as *const Tuple;

        unsafe { &*pointer }
    }

    pub fn append_element(&self, element: Term, heap: &Heap) -> &'static Tuple {
        let mut longer_element_vec: Vec<Term> = Vec::with_capacity(self.len() + 1);
        longer_element_vec.extend(self.iter());
        longer_element_vec.push(element);

        Tuple::from_slice(longer_element_vec.as_slice(), heap)
    }

    pub fn delete_element(
        &self,
        ZeroBasedIndex(index): ZeroBasedIndex,
        heap: &Heap,
    ) -> Result<&'static Tuple, Exception> {
        if index < self.len() {
            let smaller_element_vec: Vec<Term> = self
                .iter()
                .enumerate()
                .filter_map(|(old_index, old_term)| {
                    if old_index == index {
                        None
                    } else {
                        Some(old_term)
                    }
                })
                .collect();
            let smaller_tuple = Tuple::from_slice(smaller_element_vec.as_slice(), heap);

            Ok(smaller_tuple)
        } else {
            Err(badarg!())
        }
    }

    pub fn element(&self, ZeroBasedIndex(index): ZeroBasedIndex) -> exception::Result {
        if index < self.len() {
            Ok(self[index])
        } else {
            Err(badarg!())
        }
    }

    pub fn insert_element(
        &self,
        ZeroBasedIndex(index): ZeroBasedIndex,
        element: Term,
        heap: &Heap,
    ) -> Result<&'static Tuple, Exception> {
        let length = self.len();

        // can be equal to arity when insertion is at the end
        if index <= length {
            let new_arity_usize = length + 1;
            let mut larger_element_vec = Vec::with_capacity(new_arity_usize);

            for (current_index, current_element) in self.iter().enumerate() {
                if current_index == index {
                    larger_element_vec.push(element);
                }

                larger_element_vec.push(current_element);
            }

            if index == length {
                larger_element_vec.push(element);
            }

            let tuple = Tuple::from_slice(larger_element_vec.as_slice(), heap);

            Ok(tuple)
        } else {
            Err(badarg!())
        }
    }

    pub fn is_record(&self, record_tag: Term, size: Option<Term>) -> exception::Result {
        match record_tag.tag() {
            Atom => {
                let tagged = if 0 < self.len() {
                    let element = self[0];

                    match size {
                        Some(size_term) => {
                            let size_usize: usize = size_term.try_into()?;

                            (element == record_tag) & (self.len() == size_usize)
                        }
                        None => element == record_tag,
                    }
                } else {
                    // even if the `record_tag` cannot be checked, the `size` is still type checked
                    if let Some(size_term) = size {
                        let _: usize = size_term.try_into()?;
                    }

                    false
                };

                Ok(tagged.into())
            }
            _ => Err(badarg!()),
        }
    }

    pub fn iter(&self) -> Iter {
        let arity_pointer = self as *const Tuple as *const Term;
        let arity_isize = self.len() as isize;

        unsafe {
            Iter {
                pointer: arity_pointer.offset(1),
                limit: arity_pointer.offset(1 + arity_isize as isize),
            }
        }
    }

    pub fn len(&self) -> usize {
        // The `arity` field is not the same as `size` because `size` is a tagged as a small integer
        // while `arity` is tagged as an `arity` to mark the beginning of a tuple.
        unsafe { self.arity.arity_to_usize() }
    }

    pub fn setelement(
        &self,
        ZeroBasedIndex(index): ZeroBasedIndex,
        value: Term,
        heap: &Heap,
    ) -> Result<&'static Tuple, Exception> {
        let length = self.len();

        if index < length {
            let mut element_vec = Vec::with_capacity(length);

            for (current_index, current_element) in self.iter().enumerate() {
                if current_index == index {
                    element_vec.push(value);
                } else {
                    element_vec.push(current_element);
                }
            }

            let tuple = Tuple::from_slice(element_vec.as_slice(), heap);

            Ok(tuple)
        } else {
            Err(badarg!())
        }
    }

    pub fn size(&self) -> Integer {
        self.len().into()
    }

    pub fn to_list(&self, process: &Process) -> Term {
        self.iter().rfold(Term::EMPTY_LIST, |acc, element| {
            Term::cons(element, acc, &process)
        })
    }
}

impl CloneIntoHeap for &'static Tuple {
    fn clone_into_heap(&self, heap: &Heap) -> &'static Tuple {
        let arity = self.len();
        let arity_term = self.arity.clone();
        let mut term_vector = Vec::with_capacity(1 + arity);

        term_vector.push(arity_term);

        for term in self.iter() {
            term_vector.push(term.clone_into_heap(heap))
        }

        let pointer = heap.alloc_term_slice(term_vector.as_slice()) as *const Tuple;

        unsafe { &*pointer }
    }
}

#[cfg(test)]
impl Debug for Tuple {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tuple::from_slice(&[")?;

        let mut iter = self.iter();

        if let Some(first_element) = iter.next() {
            write!(f, "{:?}", first_element)?;

            for element in iter {
                write!(f, ", {:?}", element)?;
            }
        }

        write!(f, "])")
    }
}

impl Eq for Tuple {}

impl Hash for Tuple {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for element in self.iter() {
            element.hash(state);
        }
    }
}

impl Index<usize> for Tuple {
    type Output = Term;

    fn index(&self, index: usize) -> &Term {
        let length = self.len();

        assert!(index < length);

        let arity_pointer = self as *const Tuple as *const Term;
        unsafe { arity_pointer.offset(1 + index as isize).as_ref() }.unwrap()
    }
}

pub struct Iter {
    pointer: *const Term,
    limit: *const Term,
}

impl DoubleEndedIterator for Iter {
    fn next_back(&mut self) -> Option<Term> {
        if self.pointer == self.limit {
            None
        } else {
            unsafe {
                // limit is +1 past he actual elements, so pre-decrement unlike `next`, which
                // post-decrements
                self.limit = self.limit.offset(-1);
                self.limit.as_ref().map(|r| *r)
            }
        }
    }
}

impl Iterator for Iter {
    type Item = Term;

    fn next(&mut self) -> Option<Term> {
        if self.pointer == self.limit {
            None
        } else {
            let old_pointer = self.pointer;

            unsafe {
                self.pointer = self.pointer.offset(1);
                old_pointer.as_ref().map(|r| *r)
            }
        }
    }
}

impl FusedIterator for Iter {}

impl Ord for Tuple {
    fn cmp(&self, other: &Tuple) -> Ordering {
        match self.len().cmp(&other.len()) {
            Equal => self.iter().cmp(other.iter()),
            ordering => ordering,
        }
    }
}

impl PartialEq for Tuple {
    fn eq(&self, other: &Tuple) -> bool {
        (self.arity.tagged == other.arity.tagged)
            & self
                .iter()
                .zip(other.iter())
                .all(|(self_element, other_element)| self_element == other_element)
    }
}

impl PartialOrd for Tuple {
    fn partial_cmp(&self, other: &Tuple) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::process;

    mod from_slice {
        use super::*;

        use crate::process::IntoProcess;

        #[test]
        fn without_elements() {
            let process = process::local::test(&process::local::test_init());
            let tuple = Tuple::from_slice(&[], &process.heap.lock().unwrap());

            let tuple_pointer = tuple as *const Tuple;
            let arity_pointer = tuple_pointer as *const Term;
            // Avoid need to define Eq for Arity
            assert_eq!(unsafe { *arity_pointer }.tagged, Term::arity(0).tagged);
        }

        #[test]
        fn with_elements() {
            let process = process::local::test(&process::local::test_init());
            let tuple =
                Tuple::from_slice(&[0.into_process(&process)], &process.heap.lock().unwrap());

            let tuple_pointer = tuple as *const Tuple;
            let arity_pointer = tuple_pointer as *const Term;
            // Avoid need to define Eq for Arity
            assert_eq!(unsafe { *arity_pointer }.tagged, Term::arity(1).tagged);

            let element_pointer = unsafe { arity_pointer.offset(1) };
            assert_eq!(unsafe { *element_pointer }, 0.into_process(&process));
        }
    }

    mod element {
        use super::*;

        use crate::process::IntoProcess;

        #[test]
        fn without_valid_index() {
            let process = process::local::test(&process::local::test_init());
            let tuple = Tuple::from_slice(&[], &process.heap.lock().unwrap());

            assert_badarg!(tuple.element(ZeroBasedIndex(0)));
        }

        #[test]
        fn with_valid_index() {
            let process = process::local::test(&process::local::test_init());
            let tuple =
                Tuple::from_slice(&[0.into_process(&process)], &process.heap.lock().unwrap());

            assert_eq!(
                tuple.element(ZeroBasedIndex(0)),
                Ok(0.into_process(&process))
            );
        }
    }

    mod eq {
        use super::*;

        use crate::process::IntoProcess;

        #[test]
        fn without_element() {
            let process = process::local::test(&process::local::test_init());
            let tuple = Tuple::from_slice(&[], &process.heap.lock().unwrap());
            let equal = Tuple::from_slice(&[], &process.heap.lock().unwrap());

            assert_eq!(tuple, tuple);
            assert_eq!(tuple, equal);
        }

        #[test]
        fn with_unequal_length() {
            let process = process::local::test(&process::local::test_init());
            let tuple =
                Tuple::from_slice(&[0.into_process(&process)], &process.heap.lock().unwrap());
            let unequal = Tuple::from_slice(
                &[0.into_process(&process), 1.into_process(&process)],
                &process.heap.lock().unwrap(),
            );

            assert_ne!(tuple, unequal);
        }
    }

    mod iter {
        use super::*;

        use crate::process::IntoProcess;

        #[test]
        fn without_elements() {
            let process = process::local::test(&process::local::test_init());
            let tuple = Tuple::from_slice(&[], &process.heap.lock().unwrap());

            assert_eq!(tuple.iter().count(), 0);

            let size_usize: usize = tuple.size().try_into().unwrap();

            assert_eq!(tuple.iter().count(), size_usize);
        }

        #[test]
        fn with_elements() {
            let process = process::local::test(&process::local::test_init());
            let tuple =
                Tuple::from_slice(&[0.into_process(&process)], &process.heap.lock().unwrap());

            assert_eq!(tuple.iter().count(), 1);

            let size_usize: usize = tuple.size().try_into().unwrap();

            assert_eq!(tuple.iter().count(), size_usize);
        }
    }

    mod size {
        use super::*;

        use crate::process::IntoProcess;

        #[test]
        fn without_elements() {
            let process = process::local::test(&process::local::test_init());
            let tuple = Tuple::from_slice(&[], &process.heap.lock().unwrap());

            assert_eq!(tuple.size(), 0.into());
        }

        #[test]
        fn with_elements() {
            let process = process::local::test(&process::local::test_init());

            let tuple =
                Tuple::from_slice(&[0.into_process(&process)], &process.heap.lock().unwrap());

            assert_eq!(tuple.size(), 1.into());
        }
    }
}
