mod with_float;

use proptest::strategy::Just;
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::float_to_list_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_float_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_float(arc_process.clone()),
            )
        },
        |(arc_process, float)| {
            prop_assert_badarg!(
                native(&arc_process, float),
                format!("float ({}) is not a float", float)
            );

            Ok(())
        },
    );
}
