use liblumen_alloc::atom;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime;

#[native_implemented::function(erlang:display/1)]
pub fn result(term: Term) -> Term {
    runtime::sys::io::puts(&format!("{}", term));

    atom!("ok")
}
