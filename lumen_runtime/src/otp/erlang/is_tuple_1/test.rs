use proptest::prop_assert_eq;

use crate::otp::erlang::is_tuple_1::native;
use crate::test::{run, strategy};

#[test]
fn without_tuple_returns_false() {
    run(
        file!(),
        |arc_process| strategy::term::is_not_tuple(arc_process),
        |term| {
            prop_assert_eq!(native(term), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_tuple_returns_true() {
    run(
        file!(),
        |arc_process| strategy::term::tuple(arc_process.clone()),
        |term| {
            prop_assert_eq!(native(term), true.into());

            Ok(())
        },
    );
}
