use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_external_pid_left_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::pid::external(arc_process.clone()),
                    strategy::term::is_not_atom(arc_process.clone()),
                ),
                |(left, right)| {
                    prop_assert_eq!(native(left, right), true.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_external_pid_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::pid::external(arc_process.clone()),
                |operand| {
                    prop_assert_eq!(native(operand, operand), false.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_value_external_pid_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::node::external(),
                    strategy::term::pid::number(),
                    strategy::term::pid::serial(),
                )
                    .prop_map(|(arc_node, number, serial)| {
                        let mut heap = arc_process.acquire_heap();

                        (
                            heap.external_pid(arc_node.clone(), number, serial).unwrap(),
                            heap.external_pid(arc_node, number, serial).unwrap(),
                        )
                    }),
                |(left, right)| {
                    prop_assert_eq!(native(left.into(), right.into()), false.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_different_external_pid_right_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::pid::external(arc_process.clone()),
                    strategy::term::pid::external(arc_process.clone()),
                )
                    .prop_filter("Right and left must be different", |(left, right)| {
                        left != right
                    }),
                |(left, right)| {
                    prop_assert_eq!(native(left, right), true.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}
