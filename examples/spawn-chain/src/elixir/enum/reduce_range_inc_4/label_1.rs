use std::sync::Arc;

use anyhow::*;

use liblumen_alloc::erts::exception::*;
use liblumen_alloc::erts::process::frames::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::frames::{self, exception_to_native_return};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::elixir::r#enum::reduce_range_inc_4::label_2;

/// ```elixir
/// # pushed to stack: (first, last, acc, reducer)
/// # returned from call: new_first
/// # full stack: (new_first, first, last, acc, reducer)
/// # returns: new_acc
/// new_acc = reducer.(first, acc)
/// reduce_range_inc(new_first, last, new_acc, reducer)
/// ```
pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    first: Term,
    last: Term,
    acc: Term,
    reducer: Term,
) -> AllocResult<()> {
    process.stack_push(reducer)?;
    process.stack_push(acc)?;
    process.stack_push(last)?;
    process.stack_push(first)?;
    process.place_frame(frame(process), placement);

    Ok(())
}

fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    let new_first = arc_process.stack_peek(1).unwrap();
    assert!(new_first.is_integer());
    let first = arc_process.stack_peek(2).unwrap();
    assert!(first.is_integer());
    let last = arc_process.stack_peek(3).unwrap();
    assert!(last.is_integer());
    let acc = arc_process.stack_peek(4).unwrap();
    let reducer = arc_process.stack_peek(5).unwrap();

    const STACK_USED: usize = 5;

    match reducer.decode().unwrap() {
        TypedTerm::Closure(closure) => {
            arc_process.stack_popn(STACK_USED);

            label_2::place_frame_with_arguments(
                arc_process,
                Placement::Replace,
                new_first,
                last,
                reducer,
            )?;

            if closure.arity() == 2 {
                closure.place_frame_with_arguments(
                    arc_process,
                    Placement::Push,
                    vec![first, acc],
                )?;

                Process::call_native_or_yield(arc_process)
            } else {
                let argument_list = arc_process.list_from_slice(&[first, acc]).unwrap();

                exception_to_native_return(
                    arc_process,
                    STACK_USED,
                    badarity(
                        arc_process,
                        reducer,
                        argument_list,
                        anyhow!("reducer").into(),
                    ),
                )
            }
        }
        _ => exception_to_native_return(
            arc_process,
            STACK_USED,
            badfun(arc_process, reducer, anyhow!("reducer").into()),
        ),
    }
}

fn frame(process: &Process) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
