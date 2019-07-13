use super::*;

use proptest::strategy::Strategy;

#[test]
fn with_number_or_atom_second_returns_first() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::local_reference(arc_process.clone()),
                    strategy::term(arc_process.clone())
                        .prop_filter("Second must be number or atom", |second| {
                            second.is_number() || second.is_atom()
                        }),
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
fn with_lesser_local_reference_second_returns_first() {
    max(|_, process| process.reference(0).unwrap(), First);
}

#[test]
fn with_same_local_reference_second_returns_first() {
    max(|first, _| first, First);
}

#[test]
fn with_same_value_local_reference_second_returns_first() {
    max(|_, process| process.reference(1).unwrap(), First);
}

#[test]
fn with_greater_local_reference_second_returns_second() {
    max(|_, process| process.reference(2).unwrap(), Second);
}

#[test]
fn with_function_port_pid_tuple_map_list_or_bitstring_second_returns_second() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::local_reference(arc_process.clone()),
                    strategy::term(arc_process.clone()).prop_filter(
                        "Second must be function, port, pid, tuple, map, list, or bitstring",
                        |second| {
                            second.is_closure()
                                || second.is_port()
                                || second.is_pid()
                                || second.is_tuple()
                                || second.is_map()
                                || second.is_list()
                                || second.is_bitstring()
                        },
                    ),
                ),
                |(first, second)| {
                    prop_assert_eq!(erlang::max_2(first, second), second.into());

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
    super::max(|process| process.reference(1).unwrap(), second, which);
}
