use super::*;

use proptest::strategy::Strategy;

#[test]
fn with_number_atom_reference_function_port_or_pid_returns_second() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::tuple(arc_process.clone()),
                    strategy::term(arc_process.clone()).prop_filter(
                        "Second must be number, atom, reference, function, port, or pid",
                        |second| {
                            second.is_number()
                                || second.is_atom()
                                || second.is_reference()
                                || second.is_function()
                                || second.is_port()
                                || second.is_pid()
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
fn with_smaller_tuple_second_returns_second() {
    min(
        |_, process| Term::slice_to_tuple(&[1.into_process(&process)], &process),
        Second,
    );
}

#[test]
fn with_same_size_tuple_with_lesser_elements_returns_second() {
    min(
        |_, process| {
            Term::slice_to_tuple(
                &[1.into_process(&process), 1.into_process(&process)],
                &process,
            )
        },
        Second,
    );
}

#[test]
fn with_same_tuple_returns_first() {
    min(|first, _| first, First);
}

#[test]
fn with_same_value_tuple_returns_first() {
    min(
        |_, process| {
            Term::slice_to_tuple(
                &[1.into_process(&process), 2.into_process(&process)],
                &process,
            )
        },
        First,
    );
}

#[test]
fn with_same_size_tuple_with_greater_elements_returns_first() {
    min(
        |_, process| {
            Term::slice_to_tuple(
                &[1.into_process(&process), 3.into_process(&process)],
                &process,
            )
        },
        First,
    );
}

#[test]
fn with_greater_size_tuple_returns_first() {
    min(
        |_, process| {
            Term::slice_to_tuple(
                &[
                    1.into_process(&process),
                    2.into_process(&process),
                    3.into_process(&process),
                ],
                &process,
            )
        },
        First,
    );
}

#[test]
fn with_map_list_or_bitstring_second_returns_first() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::tuple(arc_process.clone()),
                    strategy::term(arc_process.clone())
                        .prop_filter("Second must be map, list, or bitstring", |second| {
                            second.is_map() || second.is_list() || second.is_bitstring()
                        }),
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
            Term::slice_to_tuple(
                &[1.into_process(&process), 2.into_process(&process)],
                &process,
            )
        },
        second,
        which,
    );
}
