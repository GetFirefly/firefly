use super::*;

#[test]
fn with_number_atom_reference_function_or_port_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::pid::local(),
                    strategy::term::number_atom_reference_function_or_port(arc_process.clone()),
                ),
                |(left, right)| {
                    prop_assert_eq!(erlang::is_greater_than_or_equal_2(left, right), true.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_greater_local_pid_right_returns_true() {
    is_greater_than_or_equal(|_, _| make_pid(0, 0).unwrap(), true);
}

#[test]
fn with_same_value_local_pid_right_returns_true() {
    is_greater_than_or_equal(|_, _| make_pid(0, 1).unwrap(), true);
}

#[test]
fn with_greater_local_pid_right_returns_false() {
    is_greater_than_or_equal(|_, _| make_pid(1, 1).unwrap(), false);
}

#[test]
fn with_external_pid_right_returns_false() {
    is_greater_than_or_equal(
        |_, process| process.external_pid_with_node_id(1, 2, 3).unwrap(),
        false,
    );
}

#[test]
fn with_list_or_bitstring_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::pid::local(),
                    strategy::term::list_or_bitstring(arc_process.clone()),
                ),
                |(left, right)| {
                    prop_assert_eq!(
                        erlang::is_greater_than_or_equal_2(left, right),
                        false.into()
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn is_greater_than_or_equal<R>(right: R, expected: bool)
where
    R: FnOnce(Term, &ProcessControlBlock) -> Term,
{
    super::is_greater_than_or_equal(|_| make_pid(0, 1).unwrap(), right, expected);
}
