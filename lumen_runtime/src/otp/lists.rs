//! Mirrors [lists](http://erlang.org/doc/man/lists.html) module

pub mod keyfind_3;
pub mod keymember_3;

use std::convert::TryInto;

use liblumen_alloc::erts::term::index::try_from_one_based_term_to_zero_based_usize;
use liblumen_alloc::erts::term::{Atom, Boxed, Cons, Term, Tuple, TypedTerm};
use liblumen_alloc::{badarg, exception};

/// Generalizes `keyfind_3`, so it can be used for `keyfind_3` or `keymember_3`
fn get_by_term_one_based_index_key(
    list: Term,
    one_based_index: Term,
    key: Term,
) -> Result<Option<Term>, exception::Exception> {
    let index_zero_based_usize: usize =
        try_from_one_based_term_to_zero_based_usize(one_based_index)?;

    get_by_zero_based_usize_index_key(list, index_zero_based_usize, key)
}

fn get_by_zero_based_usize_index_key(
    list: Term,
    zero_based_index: usize,
    key: Term,
) -> Result<Option<Term>, exception::Exception> {
    match list.to_typed_term().unwrap() {
        TypedTerm::Nil => Ok(None),
        TypedTerm::List(cons) => {
            cons_get_by_zero_based_usize_index_key(cons, zero_based_index, key)
        }
        _ => Err(badarg!().into()),
    }
}

fn cons_get_by_zero_based_usize_index_key(
    cons: Boxed<Cons>,
    zero_based_index: usize,
    key: Term,
) -> Result<Option<Term>, exception::Exception> {
    for result in cons.into_iter() {
        match result {
            Ok(list_element) => {
                let list_element_result_tuple: Result<Boxed<Tuple>, _> = list_element.try_into();

                if let Ok(list_element_tuple) = list_element_result_tuple {
                    if let Ok(list_element_tuple_element) =
                        list_element_tuple.get_element_from_zero_based_usize_index(zero_based_index)
                    {
                        if key == list_element_tuple_element {
                            return Ok(Some(list_element));
                        }
                    }
                }
            }
            Err(_) => return Err(badarg!().into()),
        }
    }

    Ok(None)
}

fn module() -> Atom {
    Atom::try_from_str("lists").unwrap()
}
