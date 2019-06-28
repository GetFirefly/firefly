use super::*;

use proptest::strategy::Strategy;

#[test]
fn with_number_or_atom_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::local_reference(arc_process.clone()),
                    strategy::term(arc_process.clone())
                        .prop_filter("Right must be number or atom", |right| {
                            right.is_number() || right.is_atom()
                        }),
                ),
                |(left, right)| {
                    prop_assert_eq!(erlang::is_less_than_2(left, right), false.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_lesser_local_reference_right_returns_false() {
    is_less_than(|_, process| Term::local_reference(0, process), false);
}

#[test]
fn with_same_local_reference_right_returns_false() {
    is_less_than(|left, _| left, false);
}

#[test]
fn with_same_value_local_reference_right_returns_false() {
    is_less_than(|_, process| Term::local_reference(1, process), false);
}

#[test]
fn with_greater_local_reference_right_returns_true() {
    is_less_than(|_, process| Term::local_reference(2, process), true);
}

#[test]
fn with_function_port_pid_tuple_map_list_or_bitstring_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::local_reference(arc_process.clone()),
                    strategy::term(arc_process.clone()).prop_filter(
                        "Right must be function, port, pid, tuple, map, list, or bitstring",
                        |right| {
                            right.is_function()
                                || right.is_port()
                                || right.is_pid()
                                || right.is_tuple()
                                || right.is_map()
                                || right.is_list()
                                || right.is_bitstring()
                        },
                    ),
                ),
                |(left, right)| {
                    prop_assert_eq!(erlang::is_less_than_2(left, right), true.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn is_less_than<R>(right: R, expected: bool)
where
    R: FnOnce(Term, &Process) -> Term,
{
    super::is_less_than(|process| Term::local_reference(1, process), right, expected);
}
