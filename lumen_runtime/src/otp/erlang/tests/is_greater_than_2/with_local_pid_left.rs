use super::*;

use proptest::prop_oneof;
use proptest::strategy::Strategy;

#[test]
fn with_number_atom_reference_function_or_port_returns_true() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (
                    strategy::term::pid::local(),
                    number_atom_reference_function_or_port(arc_process),
                )
            }),
            |(left, right)| {
                prop_assert_eq!(erlang::is_greater_than_2(left, right), true.into());

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn with_greater_local_pid_right_returns_true() {
    is_greater_than(|_, _| make_pid(0, 0).unwrap(), true);
}

#[test]
fn with_same_local_pid_right_returns_false() {
    is_greater_than(|left, _| left, false);
}

#[test]
fn with_same_value_local_pid_right_returns_false() {
    is_greater_than(|_, _| make_pid(0, 1).unwrap(), false);
}

#[test]
fn with_greater_local_pid_right_returns_false() {
    is_greater_than(|_, _| make_pid(1, 1).unwrap(), false);
}

#[test]
fn with_external_pid_right_returns_false() {
    is_greater_than(
        |_, process| process.external_pid_with_node_id(1, 2, 3).unwrap(),
        false,
    );
}

#[test]
fn with_tuple_map_list_or_bitstring_returns_false() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (
                    strategy::term::pid::local(),
                    tuple_map_list_or_bitstring(arc_process),
                )
            }),
            |(left, right)| {
                prop_assert_eq!(erlang::is_greater_than_2(left, right), false.into());

                Ok(())
            },
        )
        .unwrap();
}

fn is_greater_than<R>(right: R, expected: bool)
where
    R: FnOnce(Term, &ProcessControlBlock) -> Term,
{
    super::is_greater_than(|_| make_pid(0, 1).unwrap(), right, expected);
}

fn number_atom_reference_function_or_port(
    arc_process: Arc<ProcessControlBlock>,
) -> BoxedStrategy<Term> {
    prop_oneof![
        strategy::term::is_number(arc_process.clone()),
        strategy::term::atom(),
        strategy::term::local_reference(arc_process.clone()),
        // TODO `ExternalReference`
        strategy::term::function(arc_process),
        // TODO Port
    ]
    .boxed()
}

fn tuple_map_list_or_bitstring(arc_process: Arc<ProcessControlBlock>) -> BoxedStrategy<Term> {
    prop_oneof![
        strategy::term::tuple(arc_process.clone()),
        strategy::term::is_map(arc_process.clone()),
        strategy::term::is_list(arc_process.clone()),
        strategy::term::is_bitstring(arc_process.clone())
    ]
    .boxed()
}
