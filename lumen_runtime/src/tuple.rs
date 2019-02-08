use std::alloc::Alloc;
use std::ptr;

use crate::term::Term;

pub type Tuple = *mut Term;

pub fn slice_to_tuple(slice: &[Term], alloc: &mut Alloc) -> Tuple {
    let arity = slice.len();
    let tuple = alloc_arity(alloc, arity);

    set_arity(tuple, arity);

    for index in 0..arity {
        set_element(tuple, index, slice[index]);
    }

    tuple
}

pub fn dealloc(tuple: Tuple, alloc: &mut Alloc) {
    Term::dealloc_count(alloc, tuple, 1 + get_arity(tuple))
}

fn alloc_arity(alloc: &mut Alloc, arity: usize) -> Tuple {
    Term::alloc_count(alloc, 1 + arity)
}

fn get_arity(tuple: Tuple) -> usize {
    let arity_term = unsafe { ptr::read(tuple) };

    arity_term.into()
}

fn set_arity(tuple: Tuple, arity: usize) {
    let arity_term = Term::arity(arity);

    unsafe {
        ptr::write(tuple, arity_term);
    }
}

fn set_element(tuple: Tuple, index: usize, element: Term) {
    unsafe {
        ptr::write(tuple.offset(1 + index as isize), element);
    }
}
