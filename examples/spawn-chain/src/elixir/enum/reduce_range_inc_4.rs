mod label_1;
mod label_2;

use std::sync::Arc;

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{Atom, Term, TypedTerm};
use liblumen_alloc::erts::ModuleFunctionArity;

use lumen_runtime::otp::erlang::add_2;

/// ```elixir
/// defp reduce_range_inc(first, first, acc, fun) do
///   fun.(first, acc)
/// end
///
/// defp reduce_range_inc(first, last, acc, fun) do
///   reduce_range_inc(first + 1, last, fun.(first, acc), fun)
/// end
/// ```
pub fn place_frame_with_arguments(
    process: &Process,
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
    process.place_frame(frame(), placement);

    Ok(())
}

fn code(arc_process: &Arc<Process>) -> code::Result {
    let first = arc_process.stack_pop().unwrap();
    let last = arc_process.stack_pop().unwrap();
    let acc = arc_process.stack_pop().unwrap();
    let reducer = arc_process.stack_pop().unwrap();

    arc_process.reduce();

    // defp reduce_range_inc(first, first, acc, fun) do
    //   fun.(first, acc)
    // end
    if first == last {
        match reducer.to_typed_term().unwrap() {
            TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                TypedTerm::Closure(closure) => {
                    if closure.arity() == 2 {
                        closure.place_frame_with_arguments(
                            arc_process,
                            Placement::Replace,
                            vec![first, acc],
                        )?;

                        Process::call_code(arc_process)
                    } else {
                        let argument_list = arc_process.list_from_slice(&[first, acc])?;

                        result_from_exception(
                            arc_process,
                            liblumen_alloc::badarity!(arc_process, reducer, argument_list),
                        )
                    }
                }
                _ => result_from_exception(
                    arc_process,
                    liblumen_alloc::badfun!(arc_process, reducer),
                ),
            },
            _ => result_from_exception(arc_process, liblumen_alloc::badfun!(arc_process, reducer)),
        }
    }
    // defp reduce_range_inc(first, last, acc, fun) do
    //   reduce_range_inc(first + 1, last, fun.(first, acc), fun)
    // end
    else {
        // ```elixir
        // # pushed to stack: (first, last, acc, reducer)
        // # returned from call: new_first
        // # full stack: (new_first, first, last, acc, reducer)
        // # returns: new_acc
        // new_acc = reducer.(first, acc)
        // reduce_range_inc(new_first, last, new_acc, reducer)
        // ```
        label_1::place_frame_with_arguments(
            arc_process,
            Placement::Replace,
            first,
            last,
            acc,
            reducer,
        )?;

        // ```elixir
        // # pushed to stack: (first, inc)
        // # returned from call: N/A
        // # full stack: (first, inc)
        // # returns: new_first
        // first + 1
        // ```
        let inc = arc_process.integer(1)?;
        add_2::place_frame_with_arguments(arc_process, Placement::Push, first, inc)?;

        Process::call_code(arc_process)
    }
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("reduce_range_inc").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 3,
    })
}
