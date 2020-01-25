use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::prelude::{Boxed, Closure, Encoded, Term};

use locate_code::locate_code;

use super::frame;

/// ```elixir
/// # pushed to stack: (new_first, last, reducer)
/// # returned from call: new_acc
/// # full stack: (new_acc, last, reducer)
/// # returns: final
/// reduce_range_inc(new_first, last, new_acc, reducer)
/// ```
pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    new_first: Term,
    last: Term,
    reducer: Term,
) -> code::Result {
    assert!(new_first.is_integer());
    assert!(last.is_integer());

    let _: Boxed<Closure> = reducer.try_into().unwrap();

    process.stack_push(reducer)?;
    process.stack_push(last)?;
    process.stack_push(new_first)?;
    process.place_frame(frame(LOCATION, code), placement);

    Ok(())
}

// Private

#[locate_code]
fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    // new_acc is on top of stack because it is the return from `reducer` call
    let new_acc = arc_process.stack_pop().unwrap();
    let new_first = arc_process.stack_pop().unwrap();
    let last = arc_process.stack_pop().unwrap();
    let reducer = arc_process.stack_pop().unwrap();

    super::place_frame_with_arguments(
        arc_process,
        Placement::Replace,
        new_first,
        last,
        new_acc,
        reducer,
    )
    .unwrap();

    Process::call_code(arc_process)
}
