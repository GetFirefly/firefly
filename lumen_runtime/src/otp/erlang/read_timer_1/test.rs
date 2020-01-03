mod with_local_reference;

use proptest::strategy::Just;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang;
use crate::otp::erlang::read_timer_1::native;
use crate::process;
use crate::process::SchedulerDependentAlloc;
use crate::scheduler::with_process;
use crate::test::{has_message, receive_message, timeout_message};
use crate::test::{run, strategy};
use crate::time::Milliseconds;
use crate::timer;

#[test]
fn without_reference_errors_badarg() {
    run(
        file!(),
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_reference(arc_process.clone()),
            )
        },
        |(arc_process, timer_reference)| {
            prop_assert_badarg!(
                native(&arc_process, timer_reference),
                format!(
                    "timer_reference ({}) is not a local reference",
                    timer_reference
                )
            );

            Ok(())
        },
    );
}
