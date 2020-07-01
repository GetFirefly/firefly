use super::*;

#[test]
fn with_number_atom_reference_function_or_port_second_returns_first() {
    run!(
        |arc_process| {
            (
                strategy::term::pid::local(),
                strategy::term::number_atom_reference_function_or_port(arc_process),
            )
        },
        |(first, second)| {
            prop_assert_eq!(result(first, second), first);

            Ok(())
        },
    );
}

#[test]
fn with_lesser_local_pid_second_returns_first() {
    max(|_, _| Pid::make_term(0, 0).unwrap(), First);
}

#[test]
fn with_same_local_pid_second_returns_first() {
    max(|first, _| first, First);
}

#[test]
fn with_same_value_local_pid_second_returns_first() {
    max(|_, _| Pid::make_term(0, 1).unwrap(), First);
}

#[test]
fn with_greater_local_pid_second_returns_second() {
    max(|_, _| Pid::make_term(1, 1).unwrap(), Second);
}

#[test]
fn with_external_pid_second_returns_second() {
    max(
        |_, process| process.external_pid(external_arc_node(), 2, 3).unwrap(),
        Second,
    );
}

#[test]
fn with_list_or_bitstring_second_returns_second() {
    run!(
        |arc_process| {
            (
                strategy::term::pid::local(),
                strategy::term::tuple_map_list_or_bitstring(arc_process),
            )
        },
        |(first, second)| {
            prop_assert_eq!(result(first, second), second);

            Ok(())
        },
    );
}

fn max<R>(second: R, which: FirstSecond)
where
    R: FnOnce(Term, &Process) -> Term,
{
    super::max(|_| Pid::make_term(0, 1).unwrap(), second, which);
}
