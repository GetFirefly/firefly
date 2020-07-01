use super::*;

#[test]
fn with_number_atom_reference_function_port_or_local_pid_returns_second() {
    run!(
        |arc_process| {
            (
                strategy::term::pid::external(arc_process.clone()),
                strategy::term::number_atom_reference_function_port_or_local_pid(
                    arc_process.clone(),
                ),
            )
        },
        |(first, second)| {
            prop_assert_eq!(result(first, second), second);

            Ok(())
        },
    );
}

#[test]
fn with_lesser_external_pid_second_returns_second() {
    min(
        |_, process| process.external_pid(external_arc_node(), 1, 3).unwrap(),
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
        |_, process| process.external_pid(external_arc_node(), 2, 3).unwrap(),
        First,
    );
}

#[test]
fn with_greater_external_pid_second_returns_first() {
    min(
        |_, process| process.external_pid(external_arc_node(), 3, 3).unwrap(),
        First,
    );
}

#[test]
fn with_tuple_map_list_or_bitstring_returns_first() {
    run!(
        |arc_process| {
            (
                strategy::term::pid::external(arc_process.clone()),
                strategy::term::tuple_map_list_or_bitstring(arc_process.clone()),
            )
        },
        |(first, second)| {
            prop_assert_eq!(result(first, second), first);

            Ok(())
        },
    );
}

fn min<R>(second: R, which: FirstSecond)
where
    R: FnOnce(Term, &Process) -> Term,
{
    super::min(
        |process| process.external_pid(external_arc_node(), 2, 3).unwrap(),
        second,
        which,
    );
}
