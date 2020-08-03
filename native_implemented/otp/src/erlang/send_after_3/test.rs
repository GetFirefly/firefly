mod with_small_integer_time;

use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::term::prelude::*;

use crate::erlang;
use crate::erlang::send_after_3::result;
use crate::test;
use crate::test::strategy::milliseconds;
use crate::test::{freeze_at_timeout, freeze_timeout, has_message, registered_name, strategy};

// BigInt is not tested because it would take too long and would always count as `long_term` for the
// super short soon and later wheel sizes used for `cfg(test)`

#[test]
fn without_non_negative_integer_time_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_non_negative_integer(arc_process.clone()),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, time, message)| {
            let destination = arc_process.pid_term();

            prop_assert_badarg!(
                result(arc_process.clone(), time, destination, message),
                format!("time ({}) is not a non-negative integer", time)
            );

            Ok(())
        },
    );
}
