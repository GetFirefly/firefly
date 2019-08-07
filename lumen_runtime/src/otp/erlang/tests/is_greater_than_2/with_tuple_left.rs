use super::*;

use proptest::prop_oneof;

#[test]
fn with_number_atom_reference_function_port_or_pid_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::tuple(arc_process.clone()),
                    number_atom_reference_function_port_or_pid(arc_process.clone()),
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
        |_, process| {
            process
                .tuple_from_slice(&[process.integer(1).unwrap()])
                .unwrap()
        },
        true,
    );
}

#[test]
fn with_same_size_tuple_with_greater_elements_returns_true() {
    is_greater_than(
        |_, process| {
            process
                .tuple_from_slice(&[process.integer(1).unwrap(), process.integer(1).unwrap()])
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
                .tuple_from_slice(&[process.integer(1).unwrap(), process.integer(2).unwrap()])
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
                .tuple_from_slice(&[process.integer(1).unwrap(), process.integer(3).unwrap()])
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
                .tuple_from_slice(&[
                    process.integer(1).unwrap(),
                    process.integer(2).unwrap(),
                    process.integer(3).unwrap(),
                ])
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
                    map_list_or_bitstring(arc_process.clone()),
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
                .tuple_from_slice(&[process.integer(1).unwrap(), process.integer(2).unwrap()])
                .unwrap()
        },
        right,
        expected,
    );
}

fn number_atom_reference_function_port_or_pid(
    arc_process: Arc<ProcessControlBlock>,
) -> BoxedStrategy<Term> {
    prop_oneof![
        strategy::term::is_number(arc_process.clone()),
        strategy::term::atom(),
        strategy::term::is_reference(arc_process.clone()),
        strategy::term::function(arc_process.clone()),
        // TODO ports
        strategy::term::is_pid(arc_process)
    ]
    .boxed()
}

fn map_list_or_bitstring(arc_process: Arc<ProcessControlBlock>) -> BoxedStrategy<Term> {
    prop_oneof![
        strategy::term::map(arc_process.clone()),
        strategy::term::is_list(arc_process.clone()),
        strategy::term::is_bitstring(arc_process.clone())
    ]
    .boxed()
}
