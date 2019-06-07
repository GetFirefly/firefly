use super::*;

use proptest::strategy::Strategy;

#[test]
fn with_number_atom_reference_function_port_or_pid_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::tuple(arc_process.clone()),
                    strategy::term(arc_process.clone()).prop_filter(
                        "Right must be number, atom, reference, function, port, or pid",
                        |right| {
                            right.is_number()
                                || right.is_atom()
                                || right.is_reference()
                                || right.is_function()
                                || right.is_port()
                                || right.is_pid()
                        },
                    ),
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
fn with_smaller_tuple_right_returns_false() {
    is_less_than(
        |_, process| Term::slice_to_tuple(&[1.into_process(&process)], &process),
        false,
    );
}

#[test]
fn with_same_size_tuple_with_lesser_elements_returns_false() {
    is_less_than(
        |_, process| {
            Term::slice_to_tuple(
                &[1.into_process(&process), 1.into_process(&process)],
                &process,
            )
        },
        false,
    );
}

#[test]
fn with_same_tuple_returns_false() {
    is_less_than(|left, _| left, false);
}

#[test]
fn with_same_value_tuple_returns_false() {
    is_less_than(
        |_, process| {
            Term::slice_to_tuple(
                &[1.into_process(&process), 2.into_process(&process)],
                &process,
            )
        },
        false,
    );
}

#[test]
fn with_same_size_tuple_with_greater_elements_returns_true() {
    is_less_than(
        |_, process| {
            Term::slice_to_tuple(
                &[1.into_process(&process), 3.into_process(&process)],
                &process,
            )
        },
        true,
    );
}

#[test]
fn with_greater_size_tuple_returns_true() {
    is_less_than(
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
        true,
    );
}

#[test]
fn with_map_list_or_bitstring_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::tuple(arc_process.clone()),
                    strategy::term(arc_process.clone())
                        .prop_filter("Right must be map, list, or bitstring", |right| {
                            right.is_map() || right.is_list() || right.is_bitstring()
                        }),
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
    super::is_less_than(
        |process| {
            Term::slice_to_tuple(
                &[1.into_process(&process), 2.into_process(&process)],
                &process,
            )
        },
        right,
        expected,
    );
}
