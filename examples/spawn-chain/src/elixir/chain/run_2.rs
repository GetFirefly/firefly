//! ```elixir
//! def run(n, output) do
//!   {time, value} = :timer.tc(Chain, :create_processes, [n, output])
//!   output.("Chain.run(#{n}) in #{time} microseconds")
//!   {time, value}
//! end

mod label_1;
mod label_2;

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use liblumen_otp::timer;

// Private

#[native_implemented::function(Elixir.Chain:run/2)]
fn result(process: &Process, n: Term, output: Term) -> exception::Result<Term> {
    let module = atom!("Elixir.Chain");
    let function = atom!("create_processes");
    let arguments = process.list_from_slice(&[n, output])?;
    process.queue_frame_with_arguments(
        timer::tc_3::frame().with_arguments(false, &[module, function, arguments]),
    );

    process.queue_frame_with_arguments(label_1::frame().with_arguments(true, &[output, n]));

    Ok(Term::NONE)
}
