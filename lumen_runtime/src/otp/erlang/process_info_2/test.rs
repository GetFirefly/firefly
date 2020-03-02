mod with_local_pid;

use proptest::strategy::Just;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::process_info_2::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_local_pid_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_local_pid(arc_process.clone()),
            )
        },
        |(arc_process, pid)| {
            let item = Atom::str_to_term("registered_name");

            prop_assert_is_not_local_pid!(native(&arc_process, pid, item), pid);

            Ok(())
        },
    );
}
