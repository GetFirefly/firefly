use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{Boxed, Closure, Term, Tuple};

use crate::elixir::chain::run_2::label_2;

/// ```elixir
/// # label 1
/// # pushed to stack: (output, n)
/// # returned from call: {time, value}
/// # full stack: ({time, value}, output, n)
/// # returns: :ok
/// output.("Chain.run(#{n}) in #{time} microseconds")
/// value
pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    output: Term,
    n: Term,
) -> Result<(), Alloc> {
    assert!(output.is_function());
    assert!(n.is_integer());
    process.stack_push(n)?;
    process.stack_push(output)?;
    process.place_frame(frame(process), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let time_value = arc_process.stack_pop().unwrap();
    assert!(
        time_value.is_tuple(),
        "time_value ({:?}) isn't a tuple",
        time_value
    );
    let output = arc_process.stack_pop().unwrap();
    assert!(output.is_function());
    let n = arc_process.stack_pop().unwrap();
    assert!(n.is_integer());

    let time_value_tuple: Boxed<Tuple> = time_value.try_into().unwrap();
    assert_eq!(time_value_tuple.len(), 2);
    let time = time_value_tuple[0];
    assert!(time.is_integer());
    let value = time_value_tuple[1];
    assert!(value.is_integer());

    let output_closure: Boxed<Closure> = output.try_into().unwrap();
    assert_eq!(output_closure.arity(), 1);

    label_2::place_frame_with_arguments(arc_process, Placement::Replace, time_value)?;

    // TODO use `<>` and `to_string` to emulate interpolation more exactly
    let output_data =
        arc_process.binary_from_str(&format!("Chain.run({}) in {} microsecond(s)", n, time))?;
    output_closure.place_frame_with_arguments(arc_process, Placement::Push, vec![output_data])?;

    Process::call_code(arc_process)
}

fn frame(process: &Process) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
