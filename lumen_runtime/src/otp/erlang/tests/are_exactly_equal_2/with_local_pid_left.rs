use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_local_pid_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::pid::local(),
                    strategy::term(arc_process.clone())
                        .prop_filter("Right cannot be a local pid", |right| {
                            right.tag() != LocalPid
                        }),
                ),
                |(left, right)| {
                    prop_assert_eq!(erlang::are_exactly_equal_2(left, right), false.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_local_pid_returns_true() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(&strategy::term::pid::local(), |operand| {
            prop_assert_eq!(erlang::are_exactly_equal_2(operand, operand), true.into());

            Ok(())
        })
        .unwrap();
}

#[test]
fn with_same_value_local_pid_right_returns_true() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &(strategy::term::pid::number(), strategy::term::pid::serial()).prop_map(
                |(number, serial)| {
                    (
                        Term::local_pid(number, serial).unwrap(),
                        Term::local_pid(number, serial).unwrap(),
                    )
                },
            ),
            |(left, right)| {
                prop_assert_eq!(erlang::are_exactly_equal_2(left, right), true.into());

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn with_different_local_pid_right_returns_false() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &(strategy::term::pid::number(), strategy::term::pid::serial()).prop_map(
                |(number, serial)| {
                    (
                        Term::local_pid(number, serial).unwrap(),
                        Term::local_pid(number + 1, serial).unwrap(),
                    )
                },
            ),
            |(left, right)| {
                prop_assert_eq!(erlang::are_exactly_equal_2(left, right), false.into());

                Ok(())
            },
        )
        .unwrap();
}
