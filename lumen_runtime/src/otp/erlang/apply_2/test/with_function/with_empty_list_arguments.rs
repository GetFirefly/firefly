use super::*;

use std::sync::Arc;

use proptest::strategy::Strategy;

use liblumen_alloc::badarity;
use liblumen_alloc::erts::process::code::Code;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Atom;

use crate::test::strategy::term::export_closure;

#[test]
fn without_arity_errors_badarity() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    module_function_arity::module(),
                    module_function_arity::function(),
                    (1_u8..=255_u8),
                )
                    .prop_map(|(module, function, arity)| {
                        export_closure(&arc_process.clone(), module, function, arity)
                    }),
                |function| {
                    let Ready {
                        arc_process: child_arc_process,
                        result,
                    } = run_until_ready(
                        Default::default(),
                        |child_process| {
                            let child_function = function.clone_to_process(child_process);
                            let child_arguments = Term::NIL;

                            place_frame_with_arguments(
                                child_process,
                                Placement::Push,
                                child_function,
                                child_arguments,
                            )
                            .map_err(|e| e.into())
                        },
                        5_000,
                    )
                    .unwrap();

                    prop_assert_eq!(result, Err(badarity!(&arc_process, function, Term::NIL)));

                    mem::drop(child_arc_process);

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_arity_returns_function_return() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    module_function_arity::module(),
                    module_function_arity::function(),
                )
                    .prop_map(|(module, function)| {
                        let arity = 0;
                        let code: Code = |arc_process: &Arc<Process>| {
                            arc_process.return_from_call(Atom::str_to_term("return_from_fn"))?;

                            Process::call_code(arc_process)
                        };

                        arc_process
                            .export_closure(module, function, arity, Some(code))
                            .unwrap()
                    }),
                |function| {
                    let Ready {
                        arc_process: child_arc_process,
                        result,
                    } = run_until_ready(
                        Default::default(),
                        |child_process| {
                            let child_function = function.clone_to_process(child_process);
                            let child_arguments = Term::NIL;

                            place_frame_with_arguments(
                                child_process,
                                Placement::Push,
                                child_function,
                                child_arguments,
                            )
                            .map_err(|e| e.into())
                        },
                        5_000,
                    )
                    .unwrap();

                    prop_assert_eq!(result, Ok(Atom::str_to_term("return_from_fn")));

                    mem::drop(child_arc_process);

                    Ok(())
                },
            )
            .unwrap();
    });
}
