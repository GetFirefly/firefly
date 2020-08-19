use super::*;

#[test]
fn with_number_atom_reference_function_port_or_pid_returns_true() {
    run!(
        |arc_process| {
            (
                strategy::term::tuple(arc_process.clone()),
                strategy::term::number_atom_reference_function_port_or_pid(arc_process.clone()),
            )
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), true.into());

            Ok(())
        },
    );
}

#[test]
fn with_smaller_tuple_right_returns_true() {
    is_greater_than_or_equal(
        |_, process| process.tuple_from_slice(&[process.integer(1)]),
        true,
    );
}

#[test]
fn with_same_size_tuple_with_greater_elements_returns_true() {
    is_greater_than_or_equal(
        |_, process| process.tuple_from_slice(&[process.integer(1), process.integer(1)]),
        true,
    );
}

#[test]
fn with_same_value_tuple_returns_true() {
    is_greater_than_or_equal(
        |_, process| process.tuple_from_slice(&[process.integer(1), process.integer(2)]),
        true,
    );
}

#[test]
fn with_same_size_tuple_with_greater_elements_returns_false() {
    is_greater_than_or_equal(
        |_, process| process.tuple_from_slice(&[process.integer(1), process.integer(3)]),
        false,
    );
}

#[test]
fn with_greater_size_tuple_returns_false() {
    is_greater_than_or_equal(
        |_, process| {
            process.tuple_from_slice(&[process.integer(1), process.integer(2), process.integer(3)])
        },
        false,
    );
}

#[test]
fn with_map_list_or_bitstring_returns_false() {
    run!(
        |arc_process| {
            (
                strategy::term::tuple(arc_process.clone()),
                strategy::term::map_list_or_bitstring(arc_process.clone()),
            )
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), false.into());

            Ok(())
        },
    );
}

fn is_greater_than_or_equal<R>(right: R, expected: bool)
where
    R: FnOnce(Term, &Process) -> Term,
{
    super::is_greater_than_or_equal(
        |process| process.tuple_from_slice(&[process.integer(1), process.integer(2)]),
        right,
        expected,
    );
}
