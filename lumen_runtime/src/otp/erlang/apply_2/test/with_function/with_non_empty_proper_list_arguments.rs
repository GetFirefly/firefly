use super::*;

use std::sync::Arc;

use proptest::strategy::{Just, Strategy};

use liblumen_alloc::badarity;
use liblumen_alloc::erts::process::code::Code;
use liblumen_alloc::erts::process::Process;

use crate::test::strategy::term::export_closure;

#[test]
fn without_arity_errors_badarg() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process()
                .prop_flat_map(|arc_process| {
                    (
                        Just(arc_process.clone()),
                        module_function_arity::module(),
                        module_function_arity::function(),
                        strategy::term(arc_process.clone()),
                        strategy::term(arc_process),
                    )
                })
                .prop_map(
                    |(arc_process, module, function, first_argument, second_argument)| {
                        (
                            arc_process.clone(),
                            export_closure(&arc_process, module, function, 1),
                            arc_process
                                .list_from_slice(&[first_argument, second_argument])
                                .unwrap(),
                        )
                    },
                ),
            |(arc_process, function, arguments)| {
                let Ready {
                    arc_process: child_arc_process,
                    result,
                } = run_until_ready(
                    Default::default(),
                    |child_process| {
                        let child_function = function.clone_to_process(child_process);
                        let child_arguments = arguments.clone_to_process(child_process);

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

                prop_assert_eq!(
                    result,
                    Err(badarity!(&arc_process, function, arguments, trace()))
                );

                mem::drop(child_arc_process);

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn with_arity_returns_function_return() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    module_function_arity::module(),
                    module_function_arity::function(),
                    strategy::term(arc_process.clone()),
                )
                    .prop_map(|(module, function, argument)| {
                        let arity = 1;

                        let code: Code = |arc_process: &Arc<Process>| {
                            let return_term = arc_process.stack_pop().unwrap();
                            arc_process.return_from_call(return_term)?;

                            Process::call_code(arc_process)
                        };

                        (
                            arc_process
                                .export_closure(module, function, arity, Some(code))
                                .unwrap(),
                            argument,
                        )
                    }),
                |(function, argument)| {
                    let arguments = arc_process.list_from_slice(&[argument]).unwrap();

                    let Ready {
                        arc_process: child_arc_process,
                        result,
                    } = run_until_ready(
                        Default::default(),
                        |child_process| {
                            let child_function = function.clone_to_process(child_process);
                            let child_arguments = arguments.clone_to_process(child_process);

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

                    prop_assert_eq!(result, Ok(argument));

                    mem::drop(child_arc_process);

                    Ok(())
                },
            )
            .unwrap();
    });
}
