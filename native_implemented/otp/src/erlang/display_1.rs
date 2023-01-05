use firefly_rt::*;
use firefly_rt::term::{atoms, Term};

use crate::runtime;

#[native_implemented::function(erlang:display/1)]
pub fn result(term: Term) -> Term {
    runtime::sys::io::puts(&format!("{}", term));

    atoms::Ok.into()
}
