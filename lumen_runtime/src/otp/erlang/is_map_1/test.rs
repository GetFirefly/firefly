use proptest::prop_assert_eq;

use crate::otp::erlang::is_map_1::native;
use crate::test::strategy;

#[test]
fn without_map_returns_false() {
    run!(
        |arc_process| strategy::term::is_not_map(arc_process.clone()),
        |term| {
            prop_assert_eq!(native(term), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_map_returns_true() {
    run!(
        |arc_process| strategy::term::is_map(arc_process.clone()),
        |term| {
            prop_assert_eq!(native(term), true.into());

            Ok(())
        },
    );
}
