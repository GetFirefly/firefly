use super::*;

#[test]
fn with_number_atom_reference_function_port_or_pid_returns_second() {
    run!(
        |arc_process| {
            (
                strategy::term::tuple(arc_process.clone()),
                strategy::term::number_atom_reference_function_port_or_pid(arc_process.clone()),
            )
        },
        |(first, second)| {
            prop_assert_eq!(result(first, second), second);

            Ok(())
        },
    );
}

#[test]
fn with_smaller_tuple_second_returns_second() {
    min(
        |_, process| {
            process
                .tuple_from_slice(&[process.integer(1).unwrap()])
                .unwrap()
        },
        Second,
    );
}

#[test]
fn with_same_size_tuple_with_lesser_elements_returns_second() {
    min(
        |_, process| {
            process
                .tuple_from_slice(&[process.integer(1).unwrap(), process.integer(1).unwrap()])
                .unwrap()
        },
        Second,
    );
}

#[test]
fn with_same_tuple_returns_first() {
    min(|first, _| first, First);
}

#[test]
fn with_same_value_tuple_returns_first() {
    min(
        |_, process| {
            process
                .tuple_from_slice(&[process.integer(1).unwrap(), process.integer(2).unwrap()])
                .unwrap()
        },
        First,
    );
}

#[test]
fn with_same_size_tuple_with_greater_elements_returns_first() {
    min(
        |_, process| {
            process
                .tuple_from_slice(&[process.integer(1).unwrap(), process.integer(3).unwrap()])
                .unwrap()
        },
        First,
    );
}

#[test]
fn with_greater_size_tuple_returns_first() {
    min(
        |_, process| {
            process
                .tuple_from_slice(&[
                    process.integer(1).unwrap(),
                    process.integer(2).unwrap(),
                    process.integer(3).unwrap(),
                ])
                .unwrap()
        },
        First,
    );
}

#[test]
fn with_map_list_or_bitstring_second_returns_first() {
    run!(
        |arc_process| {
            (
                strategy::term::tuple(arc_process.clone()),
                strategy::term::map_list_or_bitstring(arc_process.clone()),
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
        |process| {
            process
                .tuple_from_slice(&[process.integer(1).unwrap(), process.integer(2).unwrap()])
                .unwrap()
        },
        second,
        which,
    );
}
