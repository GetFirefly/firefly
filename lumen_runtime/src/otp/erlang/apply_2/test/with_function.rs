mod with_empty_list_arguments;
mod with_non_empty_proper_list_arguments;

use super::*;

use crate::test::strategy::module_function_arity;

#[test]
fn without_list_arguments_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_function(arc_process.clone()),
                    strategy::term::is_not_list(arc_process.clone()),
                ),
                |(function, arguments)| {
                    let Ready {
                        arc_process: child_arc_process,
                        result,
                        ..
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

                    dbg!(&result);
                    dbg!(&arguments);

                    prop_assert_badarg!(result, format!("is not a list"));

                    mem::drop(child_arc_process);

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_list_without_proper_arguments_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_function(arc_process.clone()),
                    strategy::term::list::improper(arc_process.clone()),
                ),
                |(function, arguments)| {
                    let Ready {
                        arc_process: child_arc_process,
                        result,
                        ..
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

                    prop_assert_badarg!(
                        result,
                        format!("arguments ({}) is not a proper list", arguments)
                    );

                    mem::drop(child_arc_process);

                    Ok(())
                },
            )
            .unwrap();
    });
}
