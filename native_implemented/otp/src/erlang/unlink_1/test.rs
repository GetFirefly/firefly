mod with_local_pid;

use proptest::strategy::{Just, Strategy};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Encoded;

use crate::erlang::unlink_1::result;
use crate::test::strategy;
use crate::test::{with_process, with_process_arc};

#[test]
fn without_pid_or_port_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term(arc_process.clone())
                    .prop_filter("Cannot be pid or port", |pid_or_port| {
                        !(pid_or_port.is_pid() || pid_or_port.is_port())
                    }),
            )
        },
        |(arc_process, pid_or_port)| {
            prop_assert_badarg!(
                result(&arc_process, pid_or_port),
                format!("pid_or_port ({}) is neither a pid nor a port", pid_or_port)
            );

            Ok(())
        },
    );
}

fn link_count(process: &Process) -> usize {
    process.linked_pid_set.len()
}
