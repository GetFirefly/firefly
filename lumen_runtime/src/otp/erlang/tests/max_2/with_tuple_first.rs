use super::*;

#[test]
fn with_number_atom_reference_function_port_or_pid_returns_first() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::tuple(arc_process.clone()),
                    strategy::term::number_atom_reference_function_port_or_pid(arc_process.clone()),
                ),
                |(first, second)| {
                    prop_assert_eq!(erlang::max_2(first, second), first);

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_smaller_tuple_second_returns_first() {
    max(
        |_, process| {
            process
                .tuple_from_slice(&[process.integer(1).unwrap()])
                .unwrap()
        },
        First,
    );
}

#[test]
fn with_same_size_tuple_with_lesser_elements_returns_first() {
    max(
        |_, process| {
            process
                .tuple_from_slice(&[process.integer(1).unwrap(), process.integer(1).unwrap()])
                .unwrap()
        },
        First,
    );
}

#[test]
fn with_same_tuple_returns_first() {
    max(|first, _| first, First);
}

#[test]
fn with_same_value_tuple_returns_first() {
    max(
        |_, process| {
            process
                .tuple_from_slice(&[process.integer(1).unwrap(), process.integer(2).unwrap()])
                .unwrap()
        },
        First,
    );
}

#[test]
fn with_same_size_tuple_with_greater_elements_returns_second() {
    max(
        |_, process| {
            process
                .tuple_from_slice(&[process.integer(1).unwrap(), process.integer(3).unwrap()])
                .unwrap()
        },
        Second,
    );
}

#[test]
fn with_greater_size_tuple_returns_second() {
    max(
        |_, process| {
            process
                .tuple_from_slice(&[
                    process.integer(1).unwrap(),
                    process.integer(2).unwrap(),
                    process.integer(3).unwrap(),
                ])
                .unwrap()
        },
        Second,
    );
}

#[test]
fn with_map_list_or_bitstring_second_returns_second() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (
                    strategy::term::tuple(arc_process.clone()),
                    strategy::term::map_list_or_bitstring(arc_process.clone()),
                )
            }),
            |(first, second)| {
                prop_assert_eq!(erlang::max_2(first, second), second.into());

                Ok(())
            },
        )
        .unwrap();
}

fn max<R>(second: R, which: FirstSecond)
where
    R: FnOnce(Term, &ProcessControlBlock) -> Term,
{
    super::max(
        |process| {
            process
                .tuple_from_slice(&[process.integer(1).unwrap(), process.integer(2).unwrap()])
                .unwrap()
        },
        second,
        which,
    );
}
