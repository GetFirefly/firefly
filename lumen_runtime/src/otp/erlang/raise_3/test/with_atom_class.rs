use super::*;

use std::convert::TryInto;

use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Just, Strategy};

use liblumen_alloc::erts::exception::Class;

#[test]
fn without_class_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::atom().prop_filter(
                        "Class cannot be error, exit, or throw",
                        |class| {
                            let is_class: Result<exception::Class, _> = (*class).try_into();
                            is_class.is_err()
                        },
                    ),
                    strategy::term(arc_process.clone()),
                    strategy::term::list::proper(arc_process.clone()),
                ),
                |(class, reason, stacktrace)| {
                    prop_assert_eq!(native(class, reason, stacktrace), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_class_without_list_stacktrace_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    class(),
                    strategy::term(arc_process.clone()),
                    strategy::term::is_not_list(arc_process.clone()),
                ),
                |(class, reason, stacktrace)| {
                    prop_assert_eq!(native(class, reason, stacktrace), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_class_with_empty_list_stacktrace_raises() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    class_variant_and_term(),
                    strategy::term(arc_process.clone()),
                ),
                |((class_variant, class), reason)| {
                    let stacktrace = Term::NIL;

                    prop_assert_eq!(
                        native(class, reason, stacktrace),
                        Err(raise!(class_variant, reason, stacktrace).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_class_with_stacktrace_without_atom_module_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    class(),
                    strategy::term(arc_process.clone()),
                    strategy::term(arc_process.clone()).prop_filter(
                        "Module must not be an atom or function",
                        |module| {
                            !(
                                // {M, F, arity | args}
                                module.is_atom() ||
                                    // {function, args, location}
                                    module.is_boxed_function()
                            )
                        },
                    ),
                    strategy::term::function::function(),
                    strategy::term::function::arity_or_arguments(arc_process.clone()),
                ),
                |(class, reason, module, function, arity_or_arguments)| {
                    let stacktrace = arc_process
                        .list_from_slice(&[arc_process
                            .tuple_from_slice(&[module, function, arity_or_arguments])
                            .unwrap()])
                        .unwrap();

                    prop_assert_eq!(native(class, reason, stacktrace), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_class_with_stacktrace_with_atom_module_without_atom_function_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    class(),
                    strategy::term(arc_process.clone()),
                    strategy::term::function::module(),
                    strategy::term::is_not_atom(arc_process.clone()),
                    strategy::term::function::arity_or_arguments(arc_process.clone()),
                ),
                |(class, reason, module, function, arity_or_arguments)| {
                    let stacktrace = arc_process
                        .list_from_slice(&[arc_process
                            .tuple_from_slice(&[module, function, arity_or_arguments])
                            .unwrap()])
                        .unwrap();

                    prop_assert_eq!(native(class, reason, stacktrace), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_class_with_stacktrace_with_atom_module_with_atom_function_without_arity_or_arguments_errors_badarg(
) {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    class(),
                    strategy::term(arc_process.clone()),
                    strategy::term::function::module(),
                    strategy::term::function::function(),
                    strategy::term::function::is_not_arity_or_arguments(arc_process.clone()),
                ),
                |(class, reason, module, function, arity_or_arguments)| {
                    let stacktrace = arc_process
                        .list_from_slice(&[arc_process
                            .tuple_from_slice(&[module, function, arity_or_arguments])
                            .unwrap()])
                        .unwrap();

                    prop_assert_eq!(native(class, reason, stacktrace), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_class_with_stacktrace_with_mfa_with_file_without_charlist_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    class(),
                    strategy::term(arc_process.clone()),
                    strategy::term::function::module(),
                    strategy::term::function::function(),
                    strategy::term::function::arity_or_arguments(arc_process.clone()),
                    strategy::term::is_not_list(arc_process.clone()),
                ),
                |(class, reason, module, function, arity_or_arguments, file_value)| {
                    let file_key = atom!("file");
                    let location = arc_process
                        .list_from_slice(&[arc_process
                            .tuple_from_slice(&[file_key, file_value])
                            .unwrap()])
                        .unwrap();
                    let stacktrace = arc_process
                        .list_from_slice(&[arc_process
                            .tuple_from_slice(&[module, function, arity_or_arguments, location])
                            .unwrap()])
                        .unwrap();

                    prop_assert_eq!(native(class, reason, stacktrace), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_class_with_stacktrace_with_mfa_with_non_positive_line_with_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    class(),
                    strategy::term(arc_process.clone()),
                    strategy::term::atom(),
                    strategy::term::atom(),
                    strategy::term::function::arity_or_arguments(arc_process.clone()),
                    strategy::term::integer::non_positive(arc_process.clone()),
                ),
                |(class, reason, module, function, arity_or_arguments, line_value)| {
                    let line_key = atom!("line");
                    let location = arc_process
                        .list_from_slice(&[arc_process
                            .tuple_from_slice(&[line_key, line_value])
                            .unwrap()])
                        .unwrap();
                    let stacktrace = arc_process
                        .list_from_slice(&[arc_process
                            .tuple_from_slice(&[module, function, arity_or_arguments, location])
                            .unwrap()])
                        .unwrap();

                    prop_assert_eq!(native(class, reason, stacktrace), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_class_with_stacktrace_with_mfa_with_invalid_location_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    class(),
                    strategy::term(arc_process.clone()),
                    strategy::term::atom(),
                    strategy::term::atom(),
                    strategy::term::function::arity_or_arguments(arc_process.clone()),
                    strategy::term::atom().prop_filter("Key cannot be file or line", |key| {
                        let key_atom: Atom = (*key).try_into().unwrap();

                        match key_atom.name() {
                            "file" | "line" => false,
                            _ => true,
                        }
                    }),
                    strategy::term(arc_process.clone()),
                ),
                |(class, reason, module, function, arity_or_arguments, key, value)| {
                    let location = arc_process
                        .list_from_slice(&[arc_process.tuple_from_slice(&[key, value]).unwrap()])
                        .unwrap();
                    let stacktrace = arc_process
                        .list_from_slice(&[arc_process
                            .tuple_from_slice(&[module, function, arity_or_arguments, location])
                            .unwrap()])
                        .unwrap();

                    prop_assert_eq!(native(class, reason, stacktrace), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    })
}

#[test]
fn with_atom_module_with_atom_function_with_arity_raises() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    class_variant_and_term(),
                    strategy::term(arc_process.clone()),
                    strategy::term::function::module(),
                    strategy::term::function::function(),
                    strategy::term::integer::non_negative(arc_process.clone()),
                ),
                |((class_variant, class), reason, module, function, arity)| {
                    let stacktrace = arc_process
                        .list_from_slice(&[arc_process
                            .tuple_from_slice(&[module, function, arity])
                            .unwrap()])
                        .unwrap();

                    prop_assert_eq!(
                        native(class, reason, stacktrace),
                        Err(raise!(class_variant, reason, stacktrace).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_atom_module_with_atom_function_with_arguments_raises() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    class_variant_and_term(),
                    strategy::term(arc_process.clone()),
                    strategy::term::function::module(),
                    strategy::term::function::function(),
                    strategy::term::list::proper(arc_process.clone()),
                ),
                |((class_variant, class), reason, module, function, arguments)| {
                    let stacktrace = arc_process
                        .list_from_slice(&[arc_process
                            .tuple_from_slice(&[module, function, arguments])
                            .unwrap()])
                        .unwrap();

                    prop_assert_eq!(
                        native(class, reason, stacktrace),
                        Err(raise!(class_variant, reason, stacktrace).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_mfa_with_empty_location_raises() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    class_variant_and_term(),
                    strategy::term(arc_process.clone()),
                    strategy::term::function::module(),
                    strategy::term::function::function(),
                    strategy::term::function::arity_or_arguments(arc_process.clone()),
                ),
                |((class_variant, class), reason, module, function, arity_or_arguments)| {
                    let location = Term::NIL;
                    let stacktrace = arc_process
                        .list_from_slice(&[arc_process
                            .tuple_from_slice(&[module, function, arity_or_arguments, location])
                            .unwrap()])
                        .unwrap();

                    prop_assert_eq!(
                        native(class, reason, stacktrace),
                        Err(raise!(class_variant, reason, stacktrace).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_mfa_with_file_raises() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    class_variant_and_term(),
                    strategy::term(arc_process.clone()),
                    strategy::term::function::module(),
                    strategy::term::function::function(),
                    strategy::term::function::arity_or_arguments(arc_process.clone()),
                    strategy::term::charlist(arc_process.clone()),
                ),
                |((class_variant, class), reason, module, function, arity, file_value)| {
                    let file_key = atom!("file");
                    let location = arc_process
                        .list_from_slice(&[arc_process
                            .tuple_from_slice(&[file_key, file_value])
                            .unwrap()])
                        .unwrap();
                    let stacktrace = arc_process
                        .list_from_slice(&[arc_process
                            .tuple_from_slice(&[module, function, arity, location])
                            .unwrap()])
                        .unwrap();

                    prop_assert_eq!(
                        native(class, reason, stacktrace),
                        Err(raise!(class_variant, reason, stacktrace).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_mfa_with_positive_line_raises() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    class_variant_and_term(),
                    strategy::term(arc_process.clone()),
                    strategy::term::atom(),
                    strategy::term::atom(),
                    strategy::term::function::arity_or_arguments(arc_process.clone()),
                    strategy::term::integer::positive(arc_process.clone()),
                ),
                |(
                    (class_variant, class),
                    reason,
                    module,
                    function,
                    arity_or_arguments,
                    line_value,
                )| {
                    let line_key = atom!("line");
                    let location = arc_process
                        .list_from_slice(&[arc_process
                            .tuple_from_slice(&[line_key, line_value])
                            .unwrap()])
                        .unwrap();
                    let stacktrace = arc_process
                        .list_from_slice(&[arc_process
                            .tuple_from_slice(&[module, function, arity_or_arguments, location])
                            .unwrap()])
                        .unwrap();

                    prop_assert_eq!(
                        native(class, reason, stacktrace),
                        Err(raise!(class_variant, reason, stacktrace).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_mfa_with_file_and_line_raises() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (
                    Just(arc_process.clone()),
                    class_variant_and_term(),
                    strategy::term(arc_process.clone()),
                    strategy::term::atom(),
                    strategy::term::atom(),
                    strategy::term::function::arity_or_arguments(arc_process.clone()),
                    strategy::term::charlist(arc_process.clone()),
                    strategy::term::integer::positive(arc_process),
                )
            }),
            |(
                arc_process,
                (class_variant, class),
                reason,
                module,
                function,
                arity_or_arguments,
                file_value,
                line_value,
            )| {
                let file_key = atom!("file");
                let line_key = atom!("line");
                let location = arc_process
                    .list_from_slice(&[
                        arc_process
                            .tuple_from_slice(&[file_key, file_value])
                            .unwrap(),
                        arc_process
                            .tuple_from_slice(&[line_key, line_value])
                            .unwrap(),
                    ])
                    .unwrap();
                let stacktrace = arc_process
                    .list_from_slice(&[arc_process
                        .tuple_from_slice(&[module, function, arity_or_arguments, location])
                        .unwrap()])
                    .unwrap();

                prop_assert_eq!(
                    native(class, reason, stacktrace),
                    Err(raise!(class_variant, reason, stacktrace).into())
                );

                Ok(())
            },
        )
        .unwrap();
}

fn class() -> BoxedStrategy<Term> {
    prop_oneof![Just("error"), Just("exit"), Just("throw")]
        .prop_map(|string| atom!(&string))
        .boxed()
}

fn class_variant_and_term() -> BoxedStrategy<(Class, Term)> {
    prop_oneof![
        Just((Class::Error { arguments: None }, "error")),
        Just((Class::Exit, "exit")),
        Just((Class::Throw, "throw"))
    ]
    .prop_map(|(class_variant, string)| (class_variant, atom!(&string)))
    .boxed()
}
