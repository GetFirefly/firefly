use proptest::prop_assert_eq;

use crate::erlang::is_tuple_1::result;
use crate::test::strategy;

#[test]
fn without_tuple_returns_false() {
    run!(
        |arc_process| strategy::term::is_not_tuple(arc_process),
        |term| {
            prop_assert_eq!(result(term), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_tuple_returns_true() {
    run!(
        |arc_process| strategy::term::tuple(arc_process.clone()),
        |term| {
            prop_assert_eq!(result(term), true.into());

            Ok(())
        },
    );
}
