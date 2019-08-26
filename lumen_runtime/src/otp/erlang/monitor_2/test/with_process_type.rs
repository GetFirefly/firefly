mod with_atom_process_identifier;
mod with_local_pid_process_identifier;
mod with_tuple_process_identifier;

use super::*;

use std::convert::TryInto;

use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::term::{atom_unchecked, Atom};

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

fn is_not_process_identifier(arc_process: Arc<ProcessControlBlock>) -> BoxedStrategy<Term> {
    strategy::term(arc_process)
        .prop_filter(
            "Process identifier cannot be a pid, atom, or {atom, atom}",
            |process_identifier| match process_identifier.to_typed_term().unwrap() {
                TypedTerm::Atom(_) | TypedTerm::Pid(_) => false,
                TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                    TypedTerm::ExternalPid(_) => false,
                    TypedTerm::Tuple(tuple) => {
                        tuple.len() != 2 || !(tuple[0].is_atom() && tuple[1].is_atom())
                    }
                    _ => true,
                },
                _ => true,
            },
        )
        .boxed()
}

fn r#type() -> Term {
    atom_unchecked("process")
}
