mod with_atom_process_identifier;
mod with_local_pid_process_identifier;
mod with_tuple_process_identifier;

use super::*;

use std::convert::TryInto;

use liblumen_alloc::erts::process::code::stack::frame::Placement;

use crate::otp::erlang::{exit_1, node_0};
use crate::process;
use crate::registry;
use crate::scheduler::Scheduler;
use crate::test::{has_message, monitor_count, monitored_count, registered_name};

#[test]
fn without_process_identifier_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &is_not_process_identifier(arc_process.clone()),
                |process_identifier| {
                    prop_assert_eq!(
                        native(&arc_process, r#type(), process_identifier),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn is_not_process_identifier(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    strategy::term(arc_process)
        .prop_filter(
            "Process identifier cannot be a pid, atom, or {atom, atom}",
            |process_identifier| match process_identifier.decode().unwrap() {
                TypedTerm::Atom(_) | TypedTerm::Pid(_) => false,
                TypedTerm::ExternalPid(_) => false,
                TypedTerm::Tuple(tuple) => {
                    tuple.len() != 2 || !(tuple[0].is_atom() && tuple[1].is_atom())
                }
                _ => true,
            },
        )
        .boxed()
}

fn r#type() -> Term {
    Atom::str_to_term("process")
}
