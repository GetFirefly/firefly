use super::*;

#[test]
fn with_number_atom_reference_function_or_port_returns_false() {
    run!(
        |arc_process| {
            (
                strategy::term::pid::local(),
                strategy::term::number_atom_reference_function_or_port(arc_process),
            )
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_lesser_local_pid_right_returns_false() {
    is_less_than(|_, _| Pid::make_term(0, 0).unwrap(), false);
}

#[test]
fn with_same_local_pid_right_returns_false() {
    is_less_than(|left, _| left, false);
}

#[test]
fn with_same_value_local_pid_right_returns_false() {
    is_less_than(|_, _| Pid::make_term(0, 1).unwrap(), false);
}

#[test]
fn with_greater_local_pid_right_returns_true() {
    is_less_than(|_, _| Pid::make_term(1, 1).unwrap(), true);
}

#[test]
fn with_external_pid_right_returns_true() {
    is_less_than(
        |_, process| process.external_pid(external_arc_node(), 2, 3).unwrap(),
        true,
    );
}

#[test]
fn with_tuple_map_list_or_bitstring_returns_true() {
    run!(
        |arc_process| {
            (
                strategy::term::pid::local(),
                strategy::term::tuple_map_list_or_bitstring(arc_process),
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
    super::is_less_than(|_| Pid::make_term(0, 1).unwrap(), right, expected);
}
