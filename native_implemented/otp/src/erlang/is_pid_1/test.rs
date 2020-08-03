use proptest::prop_assert_eq;

use crate::erlang::is_pid_1::result;
use crate::test::strategy;

#[test]
fn without_pid_returns_false() {
    run!(
        |arc_process| strategy::term::is_not_pid(arc_process.clone()),
        |term| {
            prop_assert_eq!(result(term), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_pid_returns_true() {
    run!(
        |arc_process| strategy::term::is_pid(arc_process.clone()),
        |term| {
            prop_assert_eq!(result(term), true.into());

            Ok(())
        },
    );
}
