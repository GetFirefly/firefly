use std::sync::Arc;

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::{Term, TypedTerm};

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
    process: &ProcessControlBlock,
    placement: Placement,
    first: Term,
    last: Term,
    acc: Term,
    reducer: Term,
) -> Result<(), Alloc> {
    process.stack_push(reducer)?;
    process.stack_push(acc)?;
    process.stack_push(last)?;
    process.stack_push(first)?;
    process.place_frame(frame(process), placement);

    Ok(())
}

fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    arc_process.reduce();

    let new_first = arc_process.stack_pop().unwrap();
    assert!(new_first.is_integer());
    let first = arc_process.stack_pop().unwrap();
    assert!(first.is_integer());
    let last = arc_process.stack_pop().unwrap();
    assert!(last.is_integer());
    let acc = arc_process.stack_pop().unwrap();
    let reducer = arc_process.stack_pop().unwrap();

    match reducer.to_typed_term().unwrap() {
        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
            TypedTerm::Closure(closure) => {
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

                    ProcessControlBlock::call_code(arc_process)
                } else {
                    let argument_list = arc_process.list_from_slice(&[first, acc])?;

                    result_from_exception(
                        arc_process,
                        liblumen_alloc::badarity!(arc_process, reducer, argument_list,),
                    )
                }
            }
            _ => result_from_exception(arc_process, liblumen_alloc::badfun!(arc_process, reducer)),
        },
        _ => result_from_exception(arc_process, liblumen_alloc::badfun!(arc_process, reducer)),
    }
}

fn frame(process: &ProcessControlBlock) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
