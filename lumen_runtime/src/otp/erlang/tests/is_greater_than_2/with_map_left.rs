use super::*;

use proptest::strategy::Strategy;

#[test]
fn with_number_atom_reference_function_port_pid_or_tuple_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::map(arc_process.clone()),
                    strategy::term(arc_process.clone()).prop_filter(
                        "Right must be number, atom, reference, function, port, or local pid",
                        |right| {
                            right.is_number()
                                || right.is_atom()
                                || right.is_reference()
                                || right.is_function()
                                || right.is_port()
                                || right.is_pid()
                                || right.is_tuple()
                        },
                    ),
                ),
                |(left, right)| {
                    prop_assert_eq!(erlang::is_greater_than_2(left, right), true.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_smaller_map_right_returns_true() {
    is_greater_than(
        |_, process| {
            Term::slice_to_map(
                &[(
                    Term::str_to_atom("a", DoNotCare).unwrap(),
                    1.into_process(&process),
                )],
                &process,
            )
        },
        true,
    );
}

#[test]
fn with_same_size_map_with_greater_keys_returns_true() {
    is_greater_than(
        |_, process| {
            Term::slice_to_map(
                &[
                    (
                        Term::str_to_atom("a", DoNotCare).unwrap(),
                        2.into_process(&process),
                    ),
                    (
                        Term::str_to_atom("b", DoNotCare).unwrap(),
                        3.into_process(&process),
                    ),
                ],
                &process,
            )
        },
        true,
    );
}

#[test]
fn with_same_size_map_with_same_keys_with_greater_values_returns_true() {
    is_greater_than(
        |_, process| {
            Term::slice_to_map(
                &[
                    (
                        Term::str_to_atom("b", DoNotCare).unwrap(),
                        2.into_process(&process),
                    ),
                    (
                        Term::str_to_atom("c", DoNotCare).unwrap(),
                        2.into_process(&process),
                    ),
                ],
                &process,
            )
        },
        true,
    );
}

#[test]
fn with_same_map_returns_false() {
    is_greater_than(|left, _| left, false);
}

#[test]
fn with_same_value_map_returns_false() {
    is_greater_than(
        |_, process| {
            Term::slice_to_map(
                &[
                    (
                        Term::str_to_atom("b", DoNotCare).unwrap(),
                        2.into_process(&process),
                    ),
                    (
                        Term::str_to_atom("c", DoNotCare).unwrap(),
                        3.into_process(&process),
                    ),
                ],
                &process,
            )
        },
        false,
    );
}

#[test]
fn with_same_size_map_with_same_keys_with_greater_values_returns_false() {
    is_greater_than(
        |_, process| {
            Term::slice_to_map(
                &[
                    (
                        Term::str_to_atom("b", DoNotCare).unwrap(),
                        3.into_process(&process),
                    ),
                    (
                        Term::str_to_atom("c", DoNotCare).unwrap(),
                        4.into_process(&process),
                    ),
                ],
                &process,
            )
        },
        false,
    );
}

#[test]
fn with_same_size_map_with_greater_keys_returns_false() {
    is_greater_than(
        |_, process| {
            Term::slice_to_map(
                &[
                    (
                        Term::str_to_atom("c", DoNotCare).unwrap(),
                        2.into_process(&process),
                    ),
                    (
                        Term::str_to_atom("d", DoNotCare).unwrap(),
                        3.into_process(&process),
                    ),
                ],
                &process,
            )
        },
        false,
    );
}

#[test]
fn with_greater_size_map_returns_false() {
    is_greater_than(
        |_, process| {
            Term::slice_to_map(
                &[
                    (
                        Term::str_to_atom("a", DoNotCare).unwrap(),
                        1.into_process(&process),
                    ),
                    (
                        Term::str_to_atom("b", DoNotCare).unwrap(),
                        2.into_process(&process),
                    ),
                    (
                        Term::str_to_atom("c", DoNotCare).unwrap(),
                        3.into_process(&process),
                    ),
                ],
                &process,
            )
        },
        false,
    );
}

#[test]
fn with_list_or_bitstring_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::map(arc_process.clone()),
                    strategy::term(arc_process.clone()).prop_filter(
                        "Right must be number, atom, reference, function, port, or local pid",
                        |right| right.is_list() || right.is_bitstring(),
                    ),
                ),
                |(left, right)| {
                    prop_assert_eq!(erlang::is_greater_than_2(left, right), false.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn is_greater_than<R>(right: R, expected: bool)
where
    R: FnOnce(Term, &Process) -> Term,
{
    super::is_greater_than(
        |process| {
            Term::slice_to_map(
                &[
                    (
                        Term::str_to_atom("b", DoNotCare).unwrap(),
                        2.into_process(&process),
                    ),
                    (
                        Term::str_to_atom("c", DoNotCare).unwrap(),
                        3.into_process(&process),
                    ),
                ],
                &process,
            )
        },
        right,
        expected,
    );
}
