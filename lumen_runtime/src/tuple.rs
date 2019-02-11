use liblumen_arena::TypedArena;

use crate::term::Term;

pub type Tuple = *const Term;

pub fn slice_to_tuple(element_slice: &[Term], term_arena: &mut TypedArena<Term>) -> Tuple {
    let arity = element_slice.len();
    let arity_term = Term::arity(arity);
    let mut term_vector = Vec::with_capacity(1 + arity);

    term_vector.push(arity_term);
    term_vector.extend_from_slice(element_slice);

    Term::alloc_slice(term_vector.as_slice(), term_arena)
}
