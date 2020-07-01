use proptest::prop_assert;
use proptest::strategy::Just;

use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::abs_1::result;
use crate::test::strategy;

#[test]
fn without_number_errors_badarg() {
    crate::test::without_number_errors_badarg(file!(), result);
}

#[test]
fn with_number_returns_non_negative() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_number(arc_process),
            )
        },
        |(arc_process, number)| {
            let result = result(&arc_process, number);

            prop_assert!(result.is_ok());

            let abs = result.unwrap();
            let zero: Term = 0.into();

            prop_assert!(zero <= abs);

            Ok(())
        },
    );
}
