mod with_reference_timer_reference;

use std::convert::TryInto;

use proptest::strategy::Just;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::cancel_timer_2::native;
use crate::process::SchedulerDependentAlloc;
use crate::scheduler::{with_process, with_process_arc};
use crate::test::{
    cancel_timer_message, different_timeout_message, has_message, receive_message, run, strategy,
    timeout_after, timeout_after_half, timeout_after_half_and_wait, timeout_message,
    with_timer_in_different_thread, with_timer_in_same_thread,
};

#[test]
fn without_reference_timer_reference_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_reference(arc_process.clone()),
            )
        },
        |(arc_process, timer_reference)| {
            let options = Term::NIL;

            prop_assert_badarg!(
                native(&arc_process, timer_reference, options),
                format!(
                    "timer_reference ({}) is not a local reference",
                    timer_reference
                )
            );

            Ok(())
        },
    );
}

#[test]
fn with_reference_timer_reference_without_list_options_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_reference(arc_process.clone()),
                strategy::term::is_not_list(arc_process.clone()),
            )
        },
        |(arc_process, timer_reference, options)| {
            prop_assert_badarg!(
                native(&arc_process, timer_reference, options),
                "improper list"
            );

            Ok(())
        },
    );
}
