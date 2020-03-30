use proptest::prop_assert_eq;
use proptest::strategy::Just;

use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::hd_1::native;
use crate::test::strategy;

#[test]
fn without_list_errors_badarg() {
    run!(
        |arc_process| strategy::term::is_not_list(arc_process.clone()),
        |list| {
            prop_assert_is_not_non_empty_list!(native(list), list);

            Ok(())
        },
    );
}

#[test]
fn with_empty_list_errors_badarg() {
    let list = Term::NIL;

    assert_is_not_non_empty_list!(native(list), list);
}

#[test]
fn with_list_returns_head() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, head, tail)| {
            let list = arc_process.cons(head, tail).unwrap();

            prop_assert_eq!(native(list), Ok(head));

            Ok(())
        },
    );
}
