//! ```elixir
//! # label 1
//! # pushed to stack: (output, n)
//! # returned from call: {time, value}
//! # full stack: ({time, value}, output, n)
//! # returns: :ok
//! output.("Chain.run(#{n}) in #{time} microseconds")
//! value

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use super::label_2;

// Private

#[native_implemented::label]
fn result(process: &Process, time_value: Term, output: Term, n: Term) -> exception::Result<Term> {
    assert!(
        time_value.is_boxed_tuple(),
        "time_value ({:?}) isn't a tuple",
        time_value
    );
    assert!(output.is_boxed_function());
    assert!(n.is_integer());

    let time_value_tuple: Boxed<Tuple> = time_value.try_into().unwrap();
    assert_eq!(time_value_tuple.len(), 2);
    let time = time_value_tuple[0];
    assert!(time.is_integer());
    let value = time_value_tuple[1];
    assert!(value.is_integer());

    let output_closure: Boxed<Closure> = output.try_into().unwrap();
    assert_eq!(output_closure.arity(), 1);

    // TODO use `<>` and `to_string` to emulate interpolation more exactly
    let output_data =
        process.binary_from_str(&format!("Chain.run({}) in {} microsecond(s)", n, time))?;
    process
        .queue_frame_with_arguments(output_closure.frame_with_arguments(false, vec![output_data]));

    process.queue_frame_with_arguments(label_2::frame().with_arguments(true, &[time_value]));

    Ok(Term::NONE)
}
