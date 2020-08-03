use proptest::prop_assert_eq;
use proptest::strategy::Just;

use crate::erlang::tuple_to_list_1::result;
use crate::test::strategy;

#[test]
fn without_tuple_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_tuple(arc_process),
            )
        },
        |(arc_process, tuple)| {
            prop_assert_is_not_tuple!(result(&arc_process, tuple), tuple);

            Ok(())
        },
    );
}

#[test]
fn with_tuple_returns_list() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                proptest::collection::vec(strategy::term(arc_process.clone()), 0..=3),
            )
        },
        |(arc_process, element_vec)| {
            let tuple = arc_process.tuple_from_slice(&element_vec).unwrap();
            let list = arc_process.list_from_slice(&element_vec).unwrap();

            prop_assert_eq!(result(&arc_process, tuple), Ok(list));

            Ok(())
        },
    );
}
