use super::*;

use proptest::strategy::Strategy;

#[test]
fn with_number_atom_reference_function_port_or_local_pid_returns_second() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::pid::external(arc_process.clone()),
                    strategy::term(arc_process.clone()).prop_filter(
                        "Second must be number, atom, reference, function, port, or local pid",
                        |second| {
                            second.is_number()
                                || second.is_atom()
                                || second.is_reference()
                                || second.is_function()
                                || second.is_port()
                                || second.is_local_pid()
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
fn with_lesser_external_pid_second_returns_second() {
    min(
        |_, process| Term::external_pid(1, 1, 3, &process).unwrap(),
        Second,
    );
}

#[test]
fn with_same_external_pid_second_returns_first() {
    min(|first, _| first, First);
}

#[test]
fn with_same_value_external_pid_second_returns_first() {
    min(
        |_, process| Term::external_pid(1, 2, 3, &process).unwrap(),
        First,
    );
}

#[test]
fn with_greater_external_pid_second_returns_first() {
    min(
        |_, process| Term::external_pid(1, 3, 3, &process).unwrap(),
        First,
    );
}

#[test]
fn with_tuple_map_list_or_bitstring_returns_first() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::pid::external(arc_process.clone()),
                    strategy::term(arc_process.clone()).prop_filter(
                        "Second must be tuple, map, list, or bitstring",
                        |second| {
                            second.is_tuple()
                                || second.is_map()
                                || second.is_list()
                                || second.is_bitstring()
                        },
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
        |process| Term::external_pid(1, 2, 3, &process).unwrap(),
        second,
        which,
    );
}
