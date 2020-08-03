use super::*;

use proptest::prop_oneof;
use proptest::strategy::Strategy;

#[test]
fn without_non_empty_list_or_bitstring_second_returns_firsts() {
    run!(
        |arc_process| {
            strategy::term(arc_process.clone())
                .prop_filter("Second cannot be a list or bitstring", |second| {
                    !(second.is_non_empty_list() || second.is_bitstring())
                })
        },
        |second| {
            let first = Term::NIL;

            prop_assert_eq!(result(first, second), first);

            Ok(())
        },
    );
}

#[test]
fn with_non_empty_list_or_bitstring_second_returns_second() {
    run!(
        |arc_process| {
            prop_oneof![
                strategy::term::list::non_empty_maybe_improper(arc_process.clone()),
                strategy::term::is_bitstring(arc_process)
            ]
        },
        |second| {
            let first = Term::NIL;

            prop_assert_eq!(result(first, second), second);

            Ok(())
        },
    );
}
