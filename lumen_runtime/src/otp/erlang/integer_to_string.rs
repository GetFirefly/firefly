use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::erts::term::{Term, TypedTerm};

pub fn integer_to_string(integer: Term) -> Result<String, Exception> {
    let option_string: Option<String> = match integer.to_typed_term().unwrap() {
        TypedTerm::SmallInteger(small_integer) => Some(small_integer.to_string()),
        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
            TypedTerm::BigInteger(big_integer) => Some(big_integer.to_string()),
            _ => None,
        },
        _ => None,
    };

    match option_string {
        Some(string) => Ok(string),
        None => Err(badarg!().into()),
    }
}
