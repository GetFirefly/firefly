use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_external_pid_left_returns_true() {
    run!(
        |arc_process| {
            (
                strategy::term::pid::external(arc_process.clone()),
                strategy::term::is_not_atom(arc_process.clone()),
            )
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), true.into());

            Ok(())
        },
    );
}

#[test]
fn with_same_external_pid_right_returns_false() {
    run!(
        |arc_process| strategy::term::pid::external(arc_process.clone()),
        |operand| {
            prop_assert_eq!(result(operand, operand), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_same_value_external_pid_right_returns_false() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::node::external(),
                strategy::term::pid::number(),
                strategy::term::pid::serial(),
            )
                .prop_map(|(arc_process, arc_node, number, serial)| {
                    (
                        arc_process
                            .external_pid(arc_node.clone(), number, serial)
                            .unwrap(),
                        arc_process.external_pid(arc_node, number, serial).unwrap(),
                    )
                })
        },
        |(left, right)| {
            prop_assert_eq!(result(left.into(), right.into()), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_different_external_pid_right_returns_true() {
    run!(
        |arc_process| {
            (
                strategy::term::pid::external(arc_process.clone()),
                strategy::term::pid::external(arc_process.clone()),
            )
                .prop_filter("Right and left must be different", |(left, right)| {
                    left != right
                })
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), true.into());

            Ok(())
        },
    );
}
