use proptest::prop_assert_eq;

use crate::erlang::is_map_1::result;
use crate::test::strategy;

#[test]
fn without_map_returns_false() {
    run!(
        |arc_process| strategy::term::is_not_map(arc_process.clone()),
        |term| {
            prop_assert_eq!(result(term), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_map_returns_true() {
    run!(
        |arc_process| strategy::term::is_map(arc_process.clone()),
        |term| {
            prop_assert_eq!(result(term), true.into());

            Ok(())
        },
    );
}
