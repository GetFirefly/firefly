use super::*;

use proptest::strategy::Strategy;

#[test]
fn with_number_atom_reference_function_port_pid_or_tuple_second_returns_first() {
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
                                || second.is_closure()
                                || second.is_port()
                                || second.is_pid()
                                || second.is_tuple()
                        },
                    ),
                ),
                |(first, second)| {
                    prop_assert_eq!(erlang::max_2(first, second), first);

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_smaller_map_second_returns_first() {
    max(
        |_, process| {
            process
                .map_from_slice(&[(atom_unchecked("a"), process.integer(1))])
                .unwrap()
        },
        First,
    );
}

#[test]
fn with_same_size_map_with_lesser_keys_returns_first() {
    max(
        |_, process| {
            process
                .map_from_slice(&[
                    (atom_unchecked("a"), process.integer(2)),
                    (atom_unchecked("b"), process.integer(3)),
                ])
                .unwrap()
        },
        First,
    );
}

#[test]
fn with_same_size_map_with_same_keys_with_lesser_values_returns_first() {
    max(
        |_, process| {
            process
                .map_from_slice(&[
                    (atom_unchecked("b"), process.integer(2)),
                    (atom_unchecked("c"), process.integer(2)),
                ])
                .unwrap()
        },
        First,
    );
}

#[test]
fn with_same_map_returns_first() {
    max(|first, _| first, First);
}

#[test]
fn with_same_value_map_returns_first() {
    max(
        |_, process| {
            process
                .map_from_slice(&[
                    (atom_unchecked("b"), process.integer(2)),
                    (atom_unchecked("c"), process.integer(3)),
                ])
                .unwrap()
        },
        First,
    );
}

#[test]
fn with_same_size_map_with_same_keys_with_greater_values_returns_second() {
    max(
        |_, process| {
            process
                .map_from_slice(&[
                    (atom_unchecked("b"), process.integer(3)),
                    (atom_unchecked("c"), process.integer(4)),
                ])
                .unwrap()
        },
        Second,
    );
}

#[test]
fn with_same_size_map_with_greater_keys_returns_second() {
    max(
        |_, process| {
            process
                .map_from_slice(&[
                    (atom_unchecked("c"), process.integer(2)),
                    (atom_unchecked("d"), process.integer(3)),
                ])
                .unwrap()
        },
        Second,
    );
}

#[test]
fn with_greater_size_map_returns_second() {
    max(
        |_, process| {
            process
                .map_from_slice(&[
                    (atom_unchecked("a"), process.integer(1)),
                    (atom_unchecked("b"), process.integer(2)),
                    (atom_unchecked("c"), process.integer(3)),
                ])
                .unwrap()
        },
        Second,
    );
}

#[test]
fn with_map_second_returns_first() {
    max(|_, process| process.map_from_slice(&[]).unwrap(), First);
}

#[test]
fn with_list_or_bitstring_second_returns_second() {
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
                    prop_assert_eq!(erlang::max_2(first, second), second);

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn max<R>(second: R, which: FirstSecond)
where
    R: FnOnce(Term, &ProcessControlBlock) -> Term,
{
    super::max(
        |process| {
            process
                .map_from_slice(&[
                    (atom_unchecked("b"), process.integer(2)),
                    (atom_unchecked("c"), process.integer(3)),
                ])
                .unwrap()
        },
        second,
        which,
    );
}
