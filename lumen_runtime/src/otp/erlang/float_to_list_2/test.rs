mod with_float;

use proptest::strategy::Just;
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::float_to_list_2::native;
use crate::scheduler::with_process_arc;
use crate::test::{run, strategy};

#[test]
fn without_float_errors_badarg() {
    run(
        file!(),
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_float(arc_process.clone()),
            )
        },
        |(arc_process, float)| {
            let options = Term::NIL;

            prop_assert_badarg!(
                native(&arc_process, float, options),
                format!("float ({}) is not a float", float)
            );

            Ok(())
        },
    );
}
