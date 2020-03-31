mod with_local_pid;

use std::convert::TryInto;

use proptest::strategy::{BoxedStrategy, Just, Strategy};
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::registry;

use crate::erlang::process_info_2::native;
use crate::test;
use crate::test::{registered_name, strategy, with_process_arc};

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
