use proptest::prop_assert_eq;

use crate::otp::erlang::is_pid_1::native;
use crate::test::strategy;

#[test]
fn without_pid_returns_false() {
    run!(
        |arc_process| strategy::term::is_not_pid(arc_process.clone()),
        |term| {
            prop_assert_eq!(native(term), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_pid_returns_true() {
    run!(
        |arc_process| strategy::term::is_pid(arc_process.clone()),
        |term| {
            prop_assert_eq!(native(term), true.into());

            Ok(())
        },
    );
}
