use super::*;

use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Just, Strategy};
use proptest::test_runner::TestCaseError;
use firefly_rt::term::Atom;


#[test]
fn without_class_errors_badarg() {
    run!(
        |arc_process| {
            (
                strategy::term::atom().prop_filter(
                    "Class cannot be error, exit, or throw",
                    |class| {
                        let is_class: Result<exception::Class, _> = (*class).try_into();
                        is_class.is_err()
                    },
                ),
                strategy::term(arc_process.clone()),
                strategy::term::list::proper(arc_process.clone()),
            )
        },
        |(class, reason, stacktrace)| {
            prop_assert_badarg!(
                result(class, reason, stacktrace),
                "supported exception classes are error, exit, or throw"
            );

            Ok(())
        },
    );
}

#[test]
fn with_class_without_list_stacktrace_errors_badarg() {
    run!(
        |arc_process| {
            (
                class(),
                strategy::term(arc_process.clone()),
                strategy::term::is_not_list(arc_process.clone()),
            )
        },
        |(class, reason, stacktrace)| {
            prop_assert_badarg!(result(class, reason, stacktrace), "is not a list");

            Ok(())
        },
    );
}

#[test]
fn with_class_with_empty_list_stacktrace_raises() {
    run!(
        |arc_process| {
            (
                class_variant_and_term(),
                strategy::term(arc_process.clone()),
            )
        },
        |((class_variant, class), reason)| {
            let stacktrace = Term::Nil;

            prop_assert_raises(class_variant, class, reason, stacktrace)
        },
    );
}

#[test]
fn with_class_with_stacktrace_without_atom_module_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                class(),
                strategy::term(arc_process.clone()),
                strategy::term(arc_process.clone()).prop_filter(
                    "Module must not be an atom or function",
                    |module| {
                        !(
                            // {M, F, arity | args}
                            module.is_atom() ||
                                    // {function, args, location}
                                    module.is_closure()
                        )
                    },
                ),
                strategy::term::function::function(),
                strategy::term::function::arity_or_arguments(arc_process.clone()),
            )
        },
        |(arc_process, class, reason, module, function, arity_or_arguments)| {
            let stacktrace = arc_process.list_from_slice(&[arc_process.tuple_term_from_term_slice(&[
                module,
                function,
                arity_or_arguments,
            ])]);

            prop_assert_badarg!(
                result(class, reason, stacktrace),
                "{Module, Function, Arity | Args}"
            );

            Ok(())
        },
    );
}

#[test]
fn with_class_with_stacktrace_with_atom_module_without_atom_function_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                class(),
                strategy::term(arc_process.clone()),
                strategy::term::function::module(),
                strategy::term::is_not_atom(arc_process.clone()),
                strategy::term::function::arity_or_arguments(arc_process.clone()),
            )
        },
        |(arc_process, class, reason, module, function, arity_or_arguments)| {
            let stacktrace = arc_process.list_from_slice(&[arc_process.tuple_term_from_term_slice(&[
                module,
                function,
                arity_or_arguments,
            ])]);

            prop_assert_badarg!(
                result(class, reason, stacktrace),
                format!("`Function` ({}) is not an atom", function)
            );

            Ok(())
        },
    );
}

#[test]
fn with_class_with_stacktrace_with_atom_module_with_atom_function_without_arity_or_arguments_errors_badarg(
) {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                class(),
                strategy::term(arc_process.clone()),
                strategy::term::function::module(),
                strategy::term::function::function(),
                strategy::term::function::is_not_arity_or_arguments(arc_process.clone()),
            )
        },
        |(arc_process, class, reason, module, function, arity_or_arguments)| {
            let stacktrace = arc_process.list_from_slice(&[arc_process.tuple_term_from_term_slice(&[
                module,
                function,
                arity_or_arguments,
            ])]);

            prop_assert_badarg!(
                result(class, reason, stacktrace),
                "is not format `{Module, Function, Arity | Args}`"
            );

            Ok(())
        },
    );
}

#[test]
fn with_class_with_stacktrace_with_mfa_with_file_without_charlist_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                class(),
                strategy::term(arc_process.clone()),
                strategy::term::function::module(),
                strategy::term::function::function(),
                strategy::term::function::arity_or_arguments(arc_process.clone()),
                strategy::term::is_not_list(arc_process.clone()),
            )
        },
        |(arc_process, class, reason, module, function, arity_or_arguments, file_value)| {
            let file_key = atoms::File.into();
            let location = arc_process
                .list_from_slice(&[arc_process.tuple_term_from_term_slice(&[file_key, file_value])]);
            let stacktrace = arc_process.list_from_slice(&[arc_process.tuple_term_from_term_slice(&[
                module,
                function,
                arity_or_arguments,
                location,
            ])]);

            prop_assert_badarg!(
                result(class, reason, stacktrace),
                format!("file ({}) is not a non-empty list", file_value)
            );

            Ok(())
        },
    );
}

#[test]
fn with_class_with_stacktrace_with_mfa_with_non_positive_line_with_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                class(),
                strategy::term(arc_process.clone()),
                strategy::term::atom(),
                strategy::term::atom(),
                strategy::term::function::arity_or_arguments(arc_process.clone()),
                strategy::term::integer::non_positive(arc_process.clone()),
            )
        },
        |(arc_process, class, reason, module, function, arity_or_arguments, line_value)| {
            let line_key = atoms::Line.into();
            let location = arc_process
                .list_from_slice(&[arc_process.tuple_term_from_term_slice(&[line_key, line_value])]);
            let stacktrace = arc_process.list_from_slice(&[arc_process.tuple_term_from_term_slice(&[
                module,
                function,
                arity_or_arguments,
                location,
            ])]);

            prop_assert_badarg!(
                result(class, reason, stacktrace),
                format!("line ({}) is not 1 or greater", line_value)
            );

            Ok(())
        },
    );
}

#[test]
fn with_class_with_stacktrace_with_mfa_with_invalid_location_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                class(),
                strategy::term(arc_process.clone()),
                strategy::term::atom(),
                strategy::term::atom(),
                strategy::term::function::arity_or_arguments(arc_process.clone()),
                strategy::term::atom().prop_filter("Key cannot be file or line", |key| {
                    let key_atom: Atom = (*key).try_into().unwrap();

                    match key_atom.as_str() {
                        "file" | "line" => false,
                        _ => true,
                    }
                }),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, class, reason, module, function, arity_or_arguments, key, value)| {
            let location =
                arc_process.list_from_slice(&[arc_process.tuple_term_from_term_slice(&[key, value])]).unwrap();
            let stacktrace = arc_process.list_from_slice(&[arc_process.tuple_term_from_term_slice(&[
                module,
                function,
                arity_or_arguments,
                location,
            ])]);

            prop_assert_badarg!(
                result(class, reason, stacktrace),
                format!("location ({})", location)
            );

            Ok(())
        },
    );
}

#[test]
fn with_atom_module_with_atom_function_with_arity_raises() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                class_variant_and_term(),
                strategy::term(arc_process.clone()),
                strategy::term::function::module(),
                strategy::term::function::function(),
                strategy::term::integer::non_negative(arc_process.clone()),
            )
        },
        |(arc_process, (class_variant, class), reason, module, function, arity)| {
            let stacktrace = arc_process
                .list_from_slice(&[arc_process.tuple_term_from_term_slice(&[module, function, arity])]);

            prop_assert_raises(class_variant, class, reason, stacktrace)
        },
    );
}

#[test]
fn with_atom_module_with_atom_function_with_arguments_raises() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                class_variant_and_term(),
                strategy::term(arc_process.clone()),
                strategy::term::function::module(),
                strategy::term::function::function(),
                strategy::term::list::proper(arc_process.clone()),
            )
        },
        |(arc_process, (class_variant, class), reason, module, function, arguments)| {
            let stacktrace = arc_process
                .list_from_slice(&[arc_process.tuple_term_from_term_slice(&[module, function, arguments])]);

            prop_assert_raises(class_variant, class, reason, stacktrace)
        },
    );
}

#[test]
fn with_mfa_with_empty_location_raises() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                class_variant_and_term(),
                strategy::term(arc_process.clone()),
                strategy::term::function::module(),
                strategy::term::function::function(),
                strategy::term::function::arity_or_arguments(arc_process.clone()),
            )
        },
        |(arc_process, (class_variant, class), reason, module, function, arity_or_arguments)| {
            let location = Term::Nil;
            let stacktrace = arc_process.list_from_slice(&[arc_process.tuple_term_from_term_slice(&[
                module,
                function,
                arity_or_arguments,
                location,
            ])]);

            prop_assert_raises(class_variant, class, reason, stacktrace)
        },
    );
}

#[test]
fn with_mfa_with_file_raises() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                class_variant_and_term(),
                strategy::term(arc_process.clone()),
                strategy::term::function::module(),
                strategy::term::function::function(),
                strategy::term::function::arity_or_arguments(arc_process.clone()),
                strategy::term::charlist(arc_process.clone())
                    .prop_filter("File names can't be empty", |file| !file.is_nil()),
            )
        },
        |(arc_process, (class_variant, class), reason, module, function, arity, file_value)| {
            let file_key = atoms::File.into();
            let location = arc_process
                .list_from_slice(&[arc_process.tuple_term_from_term_slice(&[file_key, file_value])]);
            let stacktrace = arc_process.list_from_slice(&[
                arc_process.tuple_term_from_term_slice(&[module, function, arity, location])
            ]);

            prop_assert_raises(class_variant, class, reason, stacktrace)
        },
    );
}

#[test]
fn with_mfa_with_positive_line_raises() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                class_variant_and_term(),
                strategy::term(arc_process.clone()),
                strategy::term::atom(),
                strategy::term::atom(),
                strategy::term::function::arity_or_arguments(arc_process.clone()),
                strategy::term::integer::positive(arc_process.clone()),
            )
        },
        |(
            arc_process,
            (class_variant, class),
            reason,
            module,
            function,
            arity_or_arguments,
            line_value,
        )| {
            let line_key = atoms::Line.into();
            let location = arc_process
                .list_from_slice(&[arc_process.tuple_term_from_term_slice(&[line_key, line_value])]);
            let stacktrace = arc_process.list_from_slice(&[arc_process.tuple_term_from_term_slice(&[
                module,
                function,
                arity_or_arguments,
                location,
            ])]);

            prop_assert_raises(class_variant, class, reason, stacktrace)
        },
    );
}

#[test]
fn with_mfa_with_file_and_line_raises() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                class_variant_and_term(),
                strategy::term(arc_process.clone()),
                strategy::term::atom(),
                strategy::term::atom(),
                strategy::term::function::arity_or_arguments(arc_process.clone()),
                strategy::term::charlist(arc_process.clone())
                    .prop_filter("File names can't be empty", |file| !file.is_nil()),
                strategy::term::integer::positive(arc_process),
            )
        },
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
            let file_key = atoms::File.into();
            let line_key = atoms::Line.into();
            let location = arc_process.list_from_slice(&[
                arc_process.tuple_term_from_term_slice(&[file_key, file_value]),
                arc_process.tuple_term_from_term_slice(&[line_key, line_value]),
            ]);
            let stacktrace = arc_process.list_from_slice(&[arc_process.tuple_term_from_term_slice(&[
                module,
                function,
                arity_or_arguments,
                location,
            ])]);

            prop_assert_raises(class_variant, class, reason, stacktrace)
        },
    );
}

fn class() -> BoxedStrategy<Term> {
    prop_oneof![Just("error"), Just("exit"), Just("throw")]
        .prop_map(|string| Atom::str_to_term(&string).into())
        .boxed()
}

fn class_variant_and_term() -> BoxedStrategy<(Class, Term)> {
    prop_oneof![
        Just((Class::Error { arguments: None }, "error")),
        Just((Class::Exit, "exit")),
        Just((Class::Throw, "throw"))
    ]
    .prop_map(|(class_variant, string)| (class_variant, Atom::str_to_term(&string).into()))
    .boxed()
}

fn prop_assert_raises(
    class_variant: Class,
    class: Term,
    reason: Term,
    stacktrace: Term,
) -> Result<(), TestCaseError> {
    if let Err(Exception::Runtime(ref runtime_exception)) = result(class, reason, stacktrace) {
        prop_assert_eq!(runtime_exception.class(), class_variant);
        prop_assert_eq!(
            runtime_exception.reason(),
            reason,
            "source = {:?}",
            runtime_exception.source()
        );
        prop_assert_eq!(
            runtime_exception.stacktrace().as_term().unwrap(),
            stacktrace
        );

        Ok(())
    } else {
        Err(proptest::test_runner::TestCaseError::fail("not a raise"))
    }
}
