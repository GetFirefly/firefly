use super::*;

use proptest::strategy::Strategy;

#[test]
fn with_number_atom_reference_function_port_pid_or_tuple_second_returns_second() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::map(arc_process.clone()),
                    strategy::term(arc_process.clone()).prop_filter(
                        "Second must be number, atom, reference, function, port, or local pid",
                        |second| {
                            second.is_number()
                                || second.is_atom()
                                || second.is_reference()
                                || second.is_function()
                                || second.is_port()
                                || second.is_pid()
                                || second.is_tuple()
                        },
                    ),
                ),
                |(first, second)| {
                    prop_assert_eq!(erlang::min_2(first, second), second);

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_smaller_map_second_returns_second() {
    min(
        |_, process| {
            Term::slice_to_map(
                &[(
                    Term::str_to_atom("a", DoNotCare).unwrap(),
                    1.into_process(&process),
                )],
                &process,
            )
        },
        Second,
    );
}

#[test]
fn with_same_size_map_with_lesser_keys_returns_second() {
    min(
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
        Second,
    );
}

#[test]
fn with_same_size_map_with_same_keys_with_lesser_values_returns_second() {
    min(
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
        Second,
    );
}

#[test]
fn with_same_map_returns_first() {
    min(|first, _| first, First);
}

#[test]
fn with_same_value_map_returns_first() {
    min(
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
        First,
    );
}

#[test]
fn with_same_size_map_with_same_keys_with_greater_values_returns_first() {
    min(
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
        First,
    );
}

#[test]
fn with_same_size_map_with_greater_keys_returns_first() {
    min(
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
        First,
    );
}

#[test]
fn with_greater_size_map_returns_first() {
    min(
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
        First,
    );
}

#[test]
fn with_list_or_bitstring_second_returns_first() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::map(arc_process.clone()),
                    strategy::term(arc_process.clone()).prop_filter(
                        "Second must be number, atom, reference, function, port, or local pid",
                        |second| second.is_list() || second.is_bitstring(),
                    ),
                ),
                |(first, second)| {
                    prop_assert_eq!(erlang::min_2(first, second), first);

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn min<R>(second: R, which: FirstSecond)
where
    R: FnOnce(Term, &Process) -> Term,
{
    super::min(
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
        second,
        which,
    );
}
