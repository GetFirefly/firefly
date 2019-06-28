use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_external_pid_left_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::pid::external(arc_process.clone()),
                    strategy::term::is_not_atom(arc_process.clone()),
                ),
                |(left, right)| {
                    prop_assert_eq!(
                        erlang::are_equal_after_conversion_2(left, right),
                        false.into()
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_external_pid_right_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::pid::external(arc_process.clone()),
                |operand| {
                    prop_assert_eq!(
                        erlang::are_equal_after_conversion_2(operand, operand),
                        true.into()
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_value_external_pid_right_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::pid::external::node(),
                    strategy::term::pid::number(),
                    strategy::term::pid::serial(),
                )
                    .prop_map(|(node, number, serial)| {
                        (
                            Term::external_pid(node, number, serial, &arc_process).unwrap(),
                            Term::external_pid(node, number, serial, &arc_process).unwrap(),
                        )
                    }),
                |(left, right)| {
                    prop_assert_eq!(
                        erlang::are_equal_after_conversion_2(left, right),
                        true.into()
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_different_external_pid_right_returns_false() {
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
                    prop_assert_eq!(
                        erlang::are_equal_after_conversion_2(left, right),
                        false.into()
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
