use super::*;

#[test]
fn with_number_or_atom_second_returns_first() {
    run!(
        |arc_process| {
            (
                strategy::term::local_reference(arc_process.clone()),
                strategy::term::number_or_atom(arc_process.clone()),
            )
        },
        |(first, second)| {
            prop_assert_eq!(result(first, second), first);

            Ok(())
        },
    );
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
    run!(
        |arc_process| {
            (
                strategy::term::local_reference(arc_process.clone()),
                strategy::term::function_port_pid_tuple_map_list_or_bitstring(arc_process.clone()),
            )
        },
        |(first, second)| {
            prop_assert_eq!(result(first, second), second.into());

            Ok(())
        },
    );
}

fn max<R>(second: R, which: FirstSecond)
where
    R: FnOnce(Term, &Process) -> Term,
{
    super::max(|process| process.reference(1).unwrap(), second, which);
}
