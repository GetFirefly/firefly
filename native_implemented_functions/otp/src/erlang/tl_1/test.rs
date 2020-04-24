use proptest::prop_assert_eq;
use proptest::strategy::Just;

use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::tl_1::result;
use crate::test::strategy;

#[test]
fn without_list_errors_badarg() {
    run!(
        |arc_process| strategy::term::is_not_list(arc_process.clone()),
        |list| {
            prop_assert_is_not_non_empty_list!(result(list), list);

            Ok(())
        },
    );
}

#[test]
fn with_empty_list_errors_badarg() {
    let list = Term::NIL;

    assert_is_not_non_empty_list!(result(list), list);
}

#[test]
fn with_list_returns_tail() {
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

            prop_assert_eq!(result(list), Ok(tail));

            Ok(())
        },
    );
}
