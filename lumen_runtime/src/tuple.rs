use liblumen_arena::TypedArena;

use crate::term::Term;

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

    pub fn size(&self) -> Term {
        // The `arity` field is not the same as `size` because `size` is a tagged as a small integer
        // while `arity` is tagged as an `arity` to mark the beginning of a tuple.
        Term::arity_to_integer(&self.arity)
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
