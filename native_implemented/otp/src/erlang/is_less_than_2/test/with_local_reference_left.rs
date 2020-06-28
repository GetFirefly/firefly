use super::*;

#[test]
fn with_number_or_atom_returns_false() {
    run!(
        |arc_process| {
            (
                strategy::term::local_reference(arc_process.clone()),
                strategy::term::number_or_atom(arc_process.clone()),
            )
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_lesser_local_reference_right_returns_false() {
    is_less_than(|_, process| process.reference(0).unwrap(), false);
}

#[test]
fn with_same_local_reference_right_returns_false() {
    is_less_than(|left, _| left, false);
}

#[test]
fn with_same_value_local_reference_right_returns_false() {
    is_less_than(|_, process| process.reference(1).unwrap(), false);
}

#[test]
fn with_greater_local_reference_right_returns_true() {
    is_less_than(|_, process| process.reference(2).unwrap(), true);
}

#[test]
fn with_function_port_pid_tuple_map_list_or_bitstring_returns_true() {
    run!(
        |arc_process| {
            (
                strategy::term::local_reference(arc_process.clone()),
                strategy::term::function_port_pid_tuple_map_list_or_bitstring(arc_process.clone()),
            )
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), true.into());

            Ok(())
        },
    );
}

fn is_less_than<R>(right: R, expected: bool)
where
    R: FnOnce(Term, &Process) -> Term,
{
    super::is_less_than(|process| process.reference(1).unwrap(), right, expected);
}
