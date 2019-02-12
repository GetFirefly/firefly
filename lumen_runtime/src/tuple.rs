use std::cmp::{Eq, PartialEq};
use std::ops::Index;

use liblumen_arena::TypedArena;

use crate::term::{BadArgument, Term};

#[repr(C)]
pub struct Tuple {
    arity: Term, // the elements follow, but can't be represented in Rust
}

impl Tuple {
    pub fn from_slice<'a>(
        element_slice: &[Term],
        term_arena: &'a mut TypedArena<Term>,
    ) -> &'a Tuple {
        let arity = element_slice.len();
        let arity_term = Term::arity(arity);
        let mut term_vector = Vec::with_capacity(1 + arity);

        term_vector.push(arity_term);
        term_vector.extend_from_slice(element_slice);

        let pointer = Term::alloc_slice(term_vector.as_slice(), term_arena) as *const Tuple;

        unsafe { &*pointer }
    }

    pub fn delete_element<'a>(
        &self,
        index: usize,
        mut term_arena: &'a mut TypedArena<Term>,
    ) -> Result<&'a Tuple, BadArgument> {
        let arity_usize = usize::from(self.arity);

        if index < arity_usize {
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
            let smaller_tuple_pointer =
                Tuple::from_slice(smaller_element_vec.as_slice(), &mut term_arena) as *const Tuple;

            Ok(unsafe { &*smaller_tuple_pointer })
        } else {
            Err(BadArgument)
        }
    }

    pub fn iter(&self) -> Iter {
        let arity_pointer = self as *const Tuple as *const Term;
        let arity_isize = usize::from(self.arity) as isize;

        unsafe {
            Iter {
                pointer: arity_pointer.offset(1),
                limit: arity_pointer.offset(1 + arity_isize as isize + 1),
            }
        }
    }

    pub fn size(&self) -> Term {
        // The `arity` field is not the same as `size` because `size` is a tagged as a small integer
        // while `arity` is tagged as an `arity` to mark the beginning of a tuple.
        Term::arity_to_integer(&self.arity)
    }
}

pub trait Element<T> {
    fn element(&self, index: T) -> Result<Term, BadArgument>;
}

impl Element<usize> for Tuple {
    fn element(&self, index: usize) -> Result<Term, BadArgument> {
        let arity_usize = usize::from(self.arity);

        if index < arity_usize {
            Ok(self[index])
        } else {
            Err(BadArgument)
        }
    }
}

impl Eq for Tuple {}

impl Index<usize> for Tuple {
    type Output = Term;

    fn index(&self, index: usize) -> &Term {
        let arity_usize: usize = self.arity.into();

        assert!(index < arity_usize);

        let arity_pointer = self as *const Tuple as *const Term;
        unsafe { arity_pointer.offset(1 + index as isize).as_ref() }.unwrap()
    }
}

pub struct Iter {
    pointer: *const Term,
    limit: *const Term,
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

impl PartialEq for Tuple {
    fn eq(&self, other: &Tuple) -> bool {
        (self.arity == other.arity)
            & self
                .iter()
                .zip(other.iter())
                .all(|(self_element, other_element)| self_element == other_element)
    }

    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod from_slice {
        use super::*;

        #[test]
        fn without_elements() {
            let mut term_arena: TypedArena<Term> = Default::default();
            let tuple = Tuple::from_slice(&[], &mut term_arena);

            let tuple_pointer = tuple as *const Tuple;
            let arity_pointer = tuple_pointer as *const Term;
            assert_eq!(unsafe { *arity_pointer }, Term::arity(0));
        }

        #[test]
        fn with_elements() {
            let mut term_arena: TypedArena<Term> = Default::default();
            let tuple = Tuple::from_slice(&[0.into()], &mut term_arena);

            let tuple_pointer = tuple as *const Tuple;
            let arity_pointer = tuple_pointer as *const Term;
            assert_eq!(unsafe { *arity_pointer }, Term::arity(1));

            let element_pointer = unsafe { arity_pointer.offset(1) };
            assert_eq!(unsafe { *element_pointer }, 0.into());
        }
    }

    mod element {
        use super::*;

        #[test]
        fn without_valid_index() {
            let mut term_arena: TypedArena<Term> = Default::default();
            let tuple = Tuple::from_slice(&[], &mut term_arena);

            assert_eq!(tuple.element(0), Err(BadArgument));
        }

        #[test]
        fn with_valid_index() {
            let mut term_arena: TypedArena<Term> = Default::default();
            let tuple = Tuple::from_slice(&[0.into()], &mut term_arena);

            assert_eq!(tuple.element(0), Ok(0.into()));
        }
    }

    mod size {
        use super::*;

        #[test]
        fn without_elements() {
            let mut term_arena: TypedArena<Term> = Default::default();
            let tuple = Tuple::from_slice(&[], &mut term_arena);

            assert_eq!(tuple.size(), 0.into())
        }

        #[test]
        fn with_elements() {
            let mut term_arena: TypedArena<Term> = Default::default();
            let tuple = Tuple::from_slice(&[0.into()], &mut term_arena);

            assert_eq!(tuple.size(), 1.into())
        }
    }
}
