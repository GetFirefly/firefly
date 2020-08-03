//! ```elixir
//! # label: 4
//! # pushed to stack: ()
//! # returned from call: n
//! # full stack: (n)
//! # returns: {time, value}
//! :erlang.spawn_opt(Chain, dom, [n], [min_heap_size: 79 + n * 10])
//! ```

use std::convert::TryInto;

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use liblumen_otp::erlang;

// Private

#[native_implemented::label]
fn result(process: &Process, n: Term) -> exception::Result<Term> {
    assert!(n.is_integer());
    let n_usize: usize = n.try_into().unwrap();

    let arguments = process.list_from_slice(&[n])?;
    let min_heap_size_value = process.integer(79 + n_usize * 10)?;
    let min_heap_size_entry =
        process.tuple_from_slice(&[atom!("min_heap_size"), min_heap_size_value])?;
    let options = process.list_from_slice(&[min_heap_size_entry])?;

    process.queue_frame_with_arguments(erlang::spawn_opt_4::frame().with_arguments(
        false,
        &[atom!("Elixir.Chain"), atom!("dom"), arguments, options],
    ));

    Ok(Term::NONE)
}
