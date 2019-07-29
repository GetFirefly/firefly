use std::sync::Arc;

use liblumen_alloc::erts::process::code::stack::frame::Frame;
use liblumen_alloc::erts::process::code::{result_from_exception, Code, Result};
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::{atom_unchecked, Atom, Term, TypedTerm};
use liblumen_alloc::erts::ModuleFunctionArity;

use lumen_runtime::otp::erlang;

pub fn reduce_0_code(arc_process: &Arc<ProcessControlBlock>) -> Result {
    let enumerable = arc_process.stack_pop().unwrap();
    let initial = arc_process.stack_pop().unwrap();
    let reducer = arc_process.stack_pop().unwrap();

    match enumerable.to_typed_term().unwrap() {
        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
            TypedTerm::Map(map) => {
                match map.get(atom_unchecked("__struct__")) {
                    Some(struct_name) => {
                        if struct_name == atom_unchecked("Elixir.Range") {
                            // This assumes no one was cheeky and messed with the map
                            // representation of the struct
                            let first_key = atom_unchecked("first");
                            let first = map.get(first_key).unwrap();

                            let last_key = atom_unchecked("last");
                            let last = map.get(last_key).unwrap();

                            arc_process.reduce();

                            replace_frame_with_reduce_range_frame_with_arguments(
                                arc_process,
                                first,
                                last,
                                initial,
                                reducer,
                            )?;

                            ProcessControlBlock::call_code(arc_process)
                        } else {
                            arc_process.reduce();
                            arc_process.exception(liblumen_alloc::badarg!());

                            Ok(())
                        }
                    }
                    None => {
                        arc_process.reduce();
                        arc_process.exception(liblumen_alloc::badarg!());

                        Ok(())
                    }
                }
            }
            _ => unimplemented!(),
        },
        _ => {
            arc_process.reduce();
            arc_process.exception(liblumen_alloc::badarg!());

            Ok(())
        }
    }
}

fn replace_frame_with_reduce_range_frame_with_arguments(
    arc_process: &Arc<ProcessControlBlock>,
    first: Term,
    last: Term,
    initial: Term,
    reducer: Term,
) -> Result {
    let (function_name, code): (&str, Code) = if first <= last {
        ("reduce_range_inc", reduce_range_inc_0_code)
    } else {
        ("reduce_range_dec", reduce_range_dec_0_code)
    };

    let function = Atom::try_from_str(function_name).unwrap();
    let module_function_arity = Arc::new(ModuleFunctionArity {
        module: Atom::try_from_str("Elixir.Enum").unwrap(),
        function,
        arity: 4,
    });

    let frame = Frame::new(module_function_arity, code);
    arc_process.stack_push(reducer)?;
    arc_process.stack_push(initial)?;
    arc_process.stack_push(last)?;
    arc_process.stack_push(first)?;

    arc_process.replace_frame(frame);

    Ok(())
}

/// ```elixir
/// defp reduce_range_inc(first, first, acc, fun) do
///   fun.(first, acc)
/// end
///
/// defp reduce_range_inc(first, last, acc, fun) do
///   reduce_range_inc(first + 1, last, fun.(first, acc), fun)
/// end
/// ```
fn reduce_range_inc_0_code(arc_process: &Arc<ProcessControlBlock>) -> Result {
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
                    if closure.module_function_arity().arity == 2 {
                        arc_process.stack_push(first)?;
                        arc_process.stack_push(last)?;

                        arc_process.replace_frame(closure.frame());

                        ProcessControlBlock::call_code(arc_process)
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
        let reduce_range_inc_1_frame = Frame::new(
            arc_process.current_module_function_arity().unwrap(),
            reduce_range_inc_1_code,
        );
        arc_process.stack_push(reducer)?;
        arc_process.stack_push(acc)?;
        arc_process.stack_push(last)?;
        arc_process.stack_push(first)?;
        arc_process.replace_frame(reduce_range_inc_1_frame);

        ProcessControlBlock::call_code(arc_process)
    }
}

/// defp reduce_range_inc(first, last, acc, fun) do
///   reduce_range_inc(first + 1, last, fun.(first, acc), fun)
/// end
fn reduce_range_inc_1_code(arc_process: &Arc<ProcessControlBlock>) -> Result {
    let first = arc_process.stack_pop().unwrap();
    let last = arc_process.stack_pop().unwrap();
    let acc = arc_process.stack_pop().unwrap();
    let reducer = arc_process.stack_pop().unwrap();

    arc_process.reduce();

    match erlang::add_2(first, arc_process.integer(1)?, arc_process) {
        Ok(sum) => match reducer.to_typed_term().unwrap() {
            TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                TypedTerm::Closure(closure) => {
                    let reduce_range_inc_2_frame = Frame::new(
                        arc_process.current_module_function_arity().unwrap(),
                        reduce_range_inc_2_code,
                    );
                    arc_process.stack_push(reducer)?;
                    arc_process.stack_push(last)?;
                    arc_process.stack_push(sum)?;
                    arc_process.replace_frame(reduce_range_inc_2_frame);

                    if closure.module_function_arity().arity == 2 {
                        let function_frame = closure.frame();
                        arc_process.stack_push(first)?;
                        arc_process.stack_push(acc)?;
                        arc_process.push_frame(function_frame);

                        ProcessControlBlock::call_code(arc_process)
                    } else {
                        let argument_list = arc_process.list_from_slice(&[first, acc])?;

                        result_from_exception(
                            arc_process,
                            liblumen_alloc::badarity!(arc_process, reducer, argument_list,),
                        )
                    }
                }
                _ => result_from_exception(
                    arc_process,
                    liblumen_alloc::badfun!(arc_process, reducer),
                ),
            },
            _ => result_from_exception(arc_process, liblumen_alloc::badfun!(arc_process, reducer)),
        },
        Err(exception) => result_from_exception(arc_process, exception),
    }
}

fn reduce_range_inc_2_code(arc_process: &Arc<ProcessControlBlock>) -> Result {
    // acc is on top of stack because it is the return from `reducer` call
    let acc = arc_process.stack_pop().unwrap();
    let first = arc_process.stack_pop().unwrap();
    let last = arc_process.stack_pop().unwrap();
    let reducer = arc_process.stack_pop().unwrap();

    arc_process.reduce();

    let reduce_range_inc_0_frame = Frame::new(
        arc_process.current_module_function_arity().unwrap(),
        reduce_range_inc_0_code,
    );
    arc_process.stack_push(reducer)?;
    arc_process.stack_push(acc)?;
    arc_process.stack_push(last)?;
    arc_process.stack_push(first)?;
    arc_process.replace_frame(reduce_range_inc_0_frame);

    ProcessControlBlock::call_code(arc_process)
}

fn reduce_range_dec_0_code(_arc_process: &Arc<ProcessControlBlock>) -> Result {
    unimplemented!()
}
