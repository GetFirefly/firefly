use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_local_pid_right_returns_false() {
    run!(
        |arc_process| {
            (
                strategy::term::pid::local(),
                strategy::term(arc_process.clone())
                    .prop_filter("Right cannot be a local pid", |right| !right.is_local_pid()),
            )
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_same_local_pid_returns_true() {
    run!(|_| strategy::term::pid::local(), |operand| {
        prop_assert_eq!(result(operand, operand), true.into());

        Ok(())
    },);
}

#[test]
fn with_same_value_local_pid_right_returns_true() {
    run!(
        |_| {
            (strategy::term::pid::number(), strategy::term::pid::serial()).prop_map(
                |(number, serial)| {
                    (
                        Pid::make_term(number, serial).unwrap(),
                        Pid::make_term(number, serial).unwrap(),
                    )
                },
            )
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), true.into());

            Ok(())
        },
    );
}

#[test]
fn with_different_local_pid_right_returns_false() {
    run!(
        |_| {
            (strategy::term::pid::number(), strategy::term::pid::serial()).prop_map(
                |(number, serial)| {
                    (
                        Pid::make_term(number, serial).unwrap(),
                        Pid::make_term(number + 1, serial).unwrap(),
                    )
                },
            )
        },
        |(left, right)| {
            prop_assert_eq!(result(left, right), false.into());

            Ok(())
        },
    );
}
