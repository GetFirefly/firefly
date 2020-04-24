use super::*;

use proptest::prop_oneof;
use proptest::strategy::Strategy;

#[test]
fn without_non_empty_list_or_bitstring_returns_false() {
    run!(
        |arc_process| {
            strategy::term(arc_process.clone())
                .prop_filter("Right cannot be a list or bitstring", |right| {
                    !(right.is_list() || right.is_bitstring())
                })
        },
        |right| {
            let left = Term::NIL;

            prop_assert_eq!(result(left, right), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_non_empty_list_or_bitstring_right_returns_true() {
    run!(
        |arc_process| {
            prop_oneof![
                strategy::term::list::non_empty_maybe_improper(arc_process.clone()),
                strategy::term::is_bitstring(arc_process)
            ]
        },
        |right| {
            let left = Term::NIL;

            prop_assert_eq!(result(left, right), true.into());

            Ok(())
        },
    );
}
