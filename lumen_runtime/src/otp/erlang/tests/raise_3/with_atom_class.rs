use super::*;

use proptest::prop_oneof;
use proptest::strategy::Strategy;

use crate::exception::Class::*;

#[test]
fn without_class_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::atom().prop_filter(
                        "Class cannot be error, exit, or throw",
                        |class| match unsafe { class.atom_to_string() }.as_ref().as_ref() {
                            "error" | "exit" | "throw" => false,
                            _ => true,
                        },
                    ),
                    strategy::term(arc_process.clone()),
                    strategy::term::list::proper(arc_process.clone()),
                ),
                |(class, reason, stacktrace)| {
                    prop_assert_eq!(erlang::raise_3(class, reason, stacktrace), Err(badarg!()));

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
                    prop_assert_eq!(erlang::raise_3(class, reason, stacktrace), Err(badarg!()));

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
                    let stacktrace = Term::EMPTY_LIST;

                    prop_assert_eq!(
                        erlang::raise_3(class, reason, stacktrace),
                        Err(raise!(class_variant, reason, Some(stacktrace)))
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
                    strategy::term::is_not_atom(arc_process.clone()),
                    strategy::term::function::function(),
                    strategy::term::function::arity_or_arguments(arc_process.clone()),
                ),
                |(class, reason, module, function, arity_or_arguments)| {
                    let stacktrace = Term::slice_to_list(
                        &[Term::slice_to_tuple(
                            &[module, function, arity_or_arguments],
                            &arc_process,
                        )],
                        &arc_process,
                    );

                    prop_assert_eq!(erlang::raise_3(class, reason, stacktrace), Err(badarg!()));

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
                    let stacktrace = Term::slice_to_list(
                        &[Term::slice_to_tuple(
                            &[module, function, arity_or_arguments],
                            &arc_process,
                        )],
                        &arc_process,
                    );

                    prop_assert_eq!(erlang::raise_3(class, reason, stacktrace), Err(badarg!()));

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
                    strategy::term(arc_process.clone()).prop_filter(
                        "Arity or arguments cannot be non-negative integer or list of arguments",
                        |arity_or_arguments| {
                            !(arity_or_arguments.is_integer() || arity_or_arguments.is_list())
                        },
                    ),
                ),
                |(class, reason, module, function, arity_or_arguments)| {
                    let stacktrace = Term::slice_to_list(
                        &[Term::slice_to_tuple(
                            &[module, function, arity_or_arguments],
                            &arc_process,
                        )],
                        &arc_process,
                    );

                    prop_assert_eq!(erlang::raise_3(class, reason, stacktrace), Err(badarg!()));

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
                    let file_key = Term::str_to_atom("file", DoNotCare).unwrap();
                    let location = Term::slice_to_list(
                        &[Term::slice_to_tuple(&[file_key, file_value], &arc_process)],
                        &arc_process,
                    );
                    let stacktrace = Term::slice_to_list(
                        &[Term::slice_to_tuple(
                            &[module, function, arity_or_arguments, location],
                            &arc_process,
                        )],
                        &arc_process,
                    );

                    prop_assert_eq!(erlang::raise_3(class, reason, stacktrace), Err(badarg!()));

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
                    let line_key = Term::str_to_atom("line", DoNotCare).unwrap();
                    let location = Term::slice_to_list(
                        &[Term::slice_to_tuple(&[line_key, line_value], &arc_process)],
                        &arc_process,
                    );
                    let stacktrace = Term::slice_to_list(
                        &[Term::slice_to_tuple(
                            &[module, function, arity_or_arguments, location],
                            &arc_process,
                        )],
                        &arc_process,
                    );

                    prop_assert_eq!(erlang::raise_3(class, reason, stacktrace), Err(badarg!()));

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
                        match unsafe { key.atom_to_string() }.as_ref().as_ref() {
                            "file" | "line" => false,
                            _ => true,
                        }
                    }),
                    strategy::term(arc_process.clone()),
                ),
                |(class, reason, module, function, arity_or_arguments, key, value)| {
                    let location = Term::slice_to_list(
                        &[Term::slice_to_tuple(&[key, value], &arc_process)],
                        &arc_process,
                    );
                    let stacktrace = Term::slice_to_list(
                        &[Term::slice_to_tuple(
                            &[module, function, arity_or_arguments, location],
                            &arc_process,
                        )],
                        &arc_process,
                    );

                    prop_assert_eq!(erlang::raise_3(class, reason, stacktrace), Err(badarg!()));

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
                    let stacktrace = Term::slice_to_list(
                        &[Term::slice_to_tuple(
                            &[module, function, arity],
                            &arc_process,
                        )],
                        &arc_process,
                    );

                    prop_assert_eq!(
                        erlang::raise_3(class, reason, stacktrace),
                        Err(raise!(class_variant, reason, Some(stacktrace)))
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
                    let stacktrace = Term::slice_to_list(
                        &[Term::slice_to_tuple(
                            &[module, function, arguments],
                            &arc_process,
                        )],
                        &arc_process,
                    );

                    prop_assert_eq!(
                        erlang::raise_3(class, reason, stacktrace),
                        Err(raise!(class_variant, reason, Some(stacktrace)))
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
                    let location = Term::EMPTY_LIST;
                    let stacktrace = Term::slice_to_list(
                        &[Term::slice_to_tuple(
                            &[module, function, arity_or_arguments, location],
                            &arc_process,
                        )],
                        &arc_process,
                    );

                    prop_assert_eq!(
                        erlang::raise_3(class, reason, stacktrace),
                        Err(raise!(class_variant, reason, Some(stacktrace)))
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
                    let file_key = Term::str_to_atom("file", DoNotCare).unwrap();
                    let location = Term::slice_to_list(
                        &[Term::slice_to_tuple(&[file_key, file_value], &arc_process)],
                        &arc_process,
                    );
                    let stacktrace = Term::slice_to_list(
                        &[Term::slice_to_tuple(
                            &[module, function, arity, location],
                            &arc_process,
                        )],
                        &arc_process,
                    );

                    prop_assert_eq!(
                        erlang::raise_3(class, reason, stacktrace),
                        Err(raise!(class_variant, reason, Some(stacktrace)))
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
                    let line_key = Term::str_to_atom("line", DoNotCare).unwrap();
                    let location = Term::slice_to_list(
                        &[Term::slice_to_tuple(&[line_key, line_value], &arc_process)],
                        &arc_process,
                    );
                    let stacktrace = Term::slice_to_list(
                        &[Term::slice_to_tuple(
                            &[module, function, arity_or_arguments, location],
                            &arc_process,
                        )],
                        &arc_process,
                    );

                    prop_assert_eq!(
                        erlang::raise_3(class, reason, stacktrace),
                        Err(raise!(class_variant, reason, Some(stacktrace)))
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_mfa_with_file_and_line_raises() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    class_variant_and_term(),
                    strategy::term(arc_process.clone()),
                    strategy::term::atom(),
                    strategy::term::atom(),
                    strategy::term::function::arity_or_arguments(arc_process.clone()),
                    strategy::term::charlist(arc_process.clone()),
                    strategy::term::integer::positive(arc_process.clone()),
                ),
                |(
                    (class_variant, class),
                    reason,
                    module,
                    function,
                    arity_or_arguments,
                    file_value,
                    line_value,
                )| {
                    let file_key = Term::str_to_atom("file", DoNotCare).unwrap();
                    let line_key = Term::str_to_atom("line", DoNotCare).unwrap();
                    let location = Term::slice_to_list(
                        &[
                            Term::slice_to_tuple(&[file_key, file_value], &arc_process),
                            Term::slice_to_tuple(&[line_key, line_value], &arc_process),
                        ],
                        &arc_process,
                    );
                    let stacktrace = Term::slice_to_list(
                        &[Term::slice_to_tuple(
                            &[module, function, arity_or_arguments, location],
                            &arc_process,
                        )],
                        &arc_process,
                    );

                    prop_assert_eq!(
                        erlang::raise_3(class, reason, stacktrace),
                        Err(raise!(class_variant, reason, Some(stacktrace)))
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn class() -> BoxedStrategy<Term> {
    prop_oneof![Just("error"), Just("exit"), Just("throw")]
        .prop_map(|string| Term::str_to_atom(&string, DoNotCare).unwrap())
        .boxed()
}

fn class_variant_and_term() -> BoxedStrategy<(Class, Term)> {
    prop_oneof![
        Just((Error { arguments: None }, "error")),
        Just((Exit, "exit")),
        Just((Throw, "throw"))
    ]
    .prop_map(|(class_variant, string)| {
        (
            class_variant,
            Term::str_to_atom(&string, DoNotCare).unwrap(),
        )
    })
    .boxed()
}
