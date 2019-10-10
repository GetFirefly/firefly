use std::convert::TryInto;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::erts::term::{Term, TypedTerm};

pub fn charlist_to_string(list: Term) -> Result<String, Exception> {
    match list.to_typed_term().unwrap() {
        TypedTerm::Nil => Ok("".to_string()),
        TypedTerm::List(boxed_cons) => boxed_cons.try_into(),
        _ => Err(badarg!()),
    }
    .map_err(|runtime_exception| runtime_exception.into())
}
