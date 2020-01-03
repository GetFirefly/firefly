mod with_pid;

use anyhow::*;

use proptest::strategy::Just;
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::erts::term::prelude::Pid;

use crate::otp::erlang::is_process_alive_1::native;
use crate::scheduler::with_process_arc;
use crate::test::{run, strategy};

#[test]
fn without_pid_errors_badarg() {
    run(
        file!(),
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_pid(arc_process.clone()),
            )
        },
        |(arc_process, pid)| {
            prop_assert_is_not_local_pid!(native(&arc_process, pid), pid);

            Ok(())
        },
    );
}
