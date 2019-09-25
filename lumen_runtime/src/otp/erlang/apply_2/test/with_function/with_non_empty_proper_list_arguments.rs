use super::*;

use std::ops::RangeInclusive;
use std::sync::Arc;

use proptest::collection::SizeRange;
use proptest::strategy::{Just, Strategy};

use liblumen_alloc::badarity;
use liblumen_alloc::erts::process::code::Code;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::ModuleFunctionArity;

use crate::process::spawn::options::Options;
use crate::test::strategy::term::closure;

#[test]
fn without_arity_errors_badarg() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process()
                .prop_flat_map(|arc_process| {
                    (
                        Just(arc_process),
                        module_function_arity::module(),
                        module_function_arity::function(),
                        1_u8..=255_u8,
                    )
                })
                .prop_flat_map(|(arc_process, module, function, arity)| {
                    (
                        Just(arc_process.clone()),
                        Just(module),
                        Just(function),
                        Just(arity.clone()),
                        (Just(arc_process), module_function_arity::arity())
                            .prop_filter(
                                "Arguments arity cannot match function arity",
                                move |(_, list_arity)| *list_arity != arity,
                            )
                            .prop_flat_map(|(arc_process, list_arity)| {
                                let range_inclusive: RangeInclusive<usize> =
                                    (list_arity as usize)..=(list_arity as usize);
                                let size_range: SizeRange = range_inclusive.into();

                                (
                                    Just(arc_process.clone()),
                                    proptest::collection::vec(
                                        strategy::term(arc_process.clone()),
                                        size_range,
                                    ),
                                )
                                    .prop_map(
                                        |(arc_process, vec)| {
                                            arc_process.list_from_slice(&vec).unwrap()
                                        },
                                    )
                            }),
                    )
                })
                .prop_map(|(arc_process, module, function, arity, arguments)| {
                    (
                        arc_process.clone(),
                        closure(&arc_process.clone(), module, function, arity),
                        arguments,
                    )
                }),
            |(arc_process, function, arguments)| {
                let size_in_words = function.size_in_words() + arguments.size_in_words();
                let options = Options {
                    min_heap_size: Some(size_in_words),
                    ..Default::default()
                };

                let Ready {
                    arc_process: child_arc_process,
                    result,
                } = run_until_ready(
                    options,
                    |child_process| {
                        let child_function = function.clone_to_process(child_process);
                        let child_arguments = arguments.clone_to_process(child_process);

                        place_frame_with_arguments(
                            child_process,
                            Placement::Push,
                            child_function,
                            child_arguments,
                        )
                    },
                    5_000,
                )
                .unwrap();

                prop_assert_eq!(result, Err(badarity!(&arc_process, function, arguments)));

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
                        let creator = arc_process.pid_term();
                        let module_function_arity = Arc::new(ModuleFunctionArity {
                            module,
                            function,
                            arity: 1,
                        });
                        let code: Code = |arc_process: &Arc<Process>| {
                            let return_term = arc_process.stack_pop().unwrap();
                            arc_process.return_from_call(return_term)?;

                            Process::call_code(arc_process)
                        };

                        (
                            arc_process
                                .closure_with_env_from_slice(
                                    module_function_arity,
                                    code,
                                    creator,
                                    &[],
                                )
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
