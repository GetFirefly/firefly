mod with_proper_list_minuend;

use proptest::prop_assert_eq;
use proptest::strategy::Just;

use crate::erlang::subtract_list_2::result;
use crate::test::strategy;

#[test]
fn without_proper_list_minuend_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_proper_list(arc_process.clone()),
                strategy::term::is_list(arc_process.clone()),
            )
        },
        |(arc_process, minuend, subtrahend)| {
            prop_assert_badarg!(
                result(&arc_process, minuend, subtrahend),
                format!("minuend ({}) is not a proper list", minuend)
            );

            Ok(())
        },
    );
}
