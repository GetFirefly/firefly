use proptest::prop_assert_eq;

use crate::erlang::is_float_1::result;
use crate::test::strategy;

#[test]
fn without_float_returns_false() {
    run!(
        |arc_process| strategy::term::is_not_float(arc_process.clone()),
        |term| {
            prop_assert_eq!(result(term), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_float_returns_true() {
    run!(
        |arc_process| strategy::term::float(arc_process.clone()),
        |term| {
            prop_assert_eq!(result(term), true.into());

            Ok(())
        },
    );
}
