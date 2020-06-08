use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

use lumen_rt_full::binary_to_string::binary_to_string;
use lumen_rt_full::sys;

// Private

#[native_implemented::function(puts/1)]
fn result(elixir_string: Term) -> exception::Result<Term> {
    binary_to_string(elixir_string).map(|string| {
        // NOT A DEBUGGING LOG
        sys::io::puts(&string);

        atom!("ok")
    })
}
