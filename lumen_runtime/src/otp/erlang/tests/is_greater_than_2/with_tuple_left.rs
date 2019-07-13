use super::*;

use proptest::strategy::Strategy;

#[test]
fn with_number_atom_reference_function_port_or_pid_returns_true() {
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
                                || right.is_closure()
                                || right.is_port()
                                || right.is_pid()
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
fn with_smaller_tuple_right_returns_true() {
    is_greater_than(
        |_, process| process.tuple_from_slice(&[process.integer(1)]).unwrap(),
        true,
    );
}

#[test]
fn with_same_size_tuple_with_greater_elements_returns_true() {
    is_greater_than(
        |_, process| {
            process
                .tuple_from_slice(&[process.integer(1), process.integer(1)])
                .unwrap()
        },
        true,
    );
}

#[test]
fn with_same_tuple_returns_false() {
    is_greater_than(|left, _| left, false);
}

#[test]
fn with_same_value_tuple_returns_false() {
    is_greater_than(
        |_, process| {
            process
                .tuple_from_slice(&[process.integer(1), process.integer(2)])
                .unwrap()
        },
        false,
    );
}

#[test]
fn with_same_size_tuple_with_greater_elements_returns_false() {
    is_greater_than(
        |_, process| {
            process
                .tuple_from_slice(&[process.integer(1), process.integer(3)])
                .unwrap()
        },
        false,
    );
}

#[test]
fn with_greater_size_tuple_returns_false() {
    is_greater_than(
        |_, process| {
            process
                .tuple_from_slice(&[process.integer(1), process.integer(2), process.integer(3)])
                .unwrap()
        },
        false,
    );
}

#[test]
fn with_map_list_or_bitstring_returns_false() {
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
                    prop_assert_eq!(erlang::is_greater_than_2(left, right), false.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn is_greater_than<R>(right: R, expected: bool)
where
    R: FnOnce(Term, &ProcessControlBlock) -> Term,
{
    super::is_greater_than(
        |process| {
            process
                .tuple_from_slice(&[process.integer(1), process.integer(2)])
                .unwrap()
        },
        right,
        expected,
    );
}
