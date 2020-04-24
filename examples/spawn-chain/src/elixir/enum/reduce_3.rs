use std::sync::Arc;

use anyhow::*;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::frames::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{frames, Process};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::ModuleFunctionArity;

use crate::elixir::r#enum::reduce_range_dec_4;
use crate::elixir::r#enum::reduce_range_inc_4;

pub fn place_frame_with_arguments(
    process: &Process,
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

fn code(arc_process: &Arc<Process>) -> frames::Result {
    let enumerable = arc_process.stack_peek(1).unwrap();
    let initial = arc_process.stack_peek(2).unwrap();
    let reducer = arc_process.stack_peek(3).unwrap();

    const STACK_USED: usize = 3;

    match enumerable.decode().unwrap() {
        TypedTerm::Map(map) => {
            match map.get(Atom::str_to_term("__struct__")) {
                Some(struct_name) => {
                    if struct_name == Atom::str_to_term("Elixir.Range") {
                        // This assumes no one was cheeky and messed with the map
                        // representation of the struct
                        let first_key = Atom::str_to_term("first");
                        let first = map.get(first_key).unwrap();

                        let last_key = Atom::str_to_term("last");
                        let last = map.get(last_key).unwrap();

                        arc_process.reduce();
                        arc_process.stack_popn(STACK_USED);

                        if first <= last {
                            reduce_range_inc_4::place_frame_with_arguments(
                                arc_process,
                                Placement::Replace,
                                first,
                                last,
                                initial,
                                reducer,
                            )
                            .unwrap();
                        } else {
                            reduce_range_dec_4::place_frame_with_arguments(
                                arc_process,
                                Placement::Replace,
                                first,
                                last,
                                initial,
                                reducer,
                            )
                            .unwrap();
                        }

                        Process::call_native_or_yield(arc_process)
                    } else {
                        arc_process.reduce();
                        arc_process.stack_popn(STACK_USED);
                        arc_process.exception(
                            anyhow!("enumerable ({}) is a struct, but not a Range", enumerable)
                                .into(),
                        );

                        Ok(())
                    }
                }
                None => {
                    arc_process.reduce();
                    arc_process.stack_popn(STACK_USED);
                    arc_process.exception(
                        anyhow!("enumerable ({}) is a map, but not a struct", enumerable).into(),
                    );

                    Ok(())
                }
            }
        }
        _ => {
            arc_process.reduce();
            arc_process.stack_popn(STACK_USED);
            arc_process.exception(anyhow!("enumerable ({}) is not a map", enumerable).into());

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
