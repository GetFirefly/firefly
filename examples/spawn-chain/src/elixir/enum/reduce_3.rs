use std::sync::Arc;

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, ProcessControlBlock};
use liblumen_alloc::erts::term::{atom_unchecked, Atom, Term, TypedTerm};
use liblumen_alloc::erts::ModuleFunctionArity;

use crate::elixir::r#enum::reduce_range_dec_4;
use crate::elixir::r#enum::reduce_range_inc_4;

pub fn place_frame_with_arguments(
    process: &ProcessControlBlock,
    placement: Placement,
    enumerable: Term,
    acc: Term,
    reducer: Term,
) -> Result<(), Alloc> {
    process.stack_push(reducer)?;
    process.stack_push(acc)?;
    process.stack_push(enumerable)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
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

                            if first <= last {
                                reduce_range_inc_4::place_frame_with_arguments(
                                    arc_process,
                                    Placement::Replace,
                                    first,
                                    last,
                                    initial,
                                    reducer,
                                )?;
                            } else {
                                reduce_range_dec_4::place_frame_with_arguments(
                                    arc_process,
                                    Placement::Replace,
                                    first,
                                    last,
                                    initial,
                                    reducer,
                                )?;
                            }

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

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("reduce").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 3,
    })
}
