use super::*;

use proptest::strategy::Strategy;

#[test]
fn with_number_atom_reference_function_or_port_second_returns_first() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::pid::local(),
                    strategy::term(arc_process.clone()).prop_filter(
                        "Second must be number, atom, reference, function, or port",
                        |second| {
                            second.is_number()
                                || second.is_atom()
                                || second.is_reference()
                                || second.is_closure()
                                || second.is_port()
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
fn with_lesser_local_pid_second_returns_first() {
    max(|_, _| make_pid(0, 0).unwrap(), First);
}

#[test]
fn with_same_local_pid_second_returns_first() {
    max(|first, _| first, First);
}

#[test]
fn with_same_value_local_pid_second_returns_first() {
    max(|_, _| make_pid(0, 1).unwrap(), First);
}

#[test]
fn with_greater_local_pid_second_returns_second() {
    max(|_, _| make_pid(1, 1).unwrap(), Second);
}

#[test]
fn with_external_pid_second_returns_second() {
    max(
        |_, process| process.external_pid_with_node_id(1, 2, 3).unwrap(),
        Second,
    );
}

#[test]
fn with_list_or_bitstring_second_returns_second() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::pid::local(),
                    strategy::term(arc_process.clone())
                        .prop_filter("second must be tuple, map, list, or bitstring", |second| {
                            second.is_list() || second.is_bitstring()
                        }),
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
    super::max(|_| make_pid(0, 1).unwrap(), second, which);
}
