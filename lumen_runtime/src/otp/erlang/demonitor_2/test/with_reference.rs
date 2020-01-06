mod with_flush_and_info_options;
mod with_flush_option;
mod with_info_option;
mod without_options;

use super::*;

use std::sync::Arc;

use liblumen_alloc::atom;
use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::term::prelude::Atom;

use crate::otp::erlang::exit_1;
use crate::process;
use crate::scheduler::Scheduler;
use crate::test::{has_message, monitor_count, monitored_count};

#[test]
fn without_proper_list_for_options_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_reference(arc_process.clone()),
                strategy::term::is_not_list(arc_process.clone()),
            )
        },
        |(arc_process, reference, tail)| {
            let options = arc_process
                .improper_list_from_slice(&[atom!("flush")], tail)
                .unwrap();

            prop_assert_badarg!(native(&arc_process, reference, options), "improper list");

            Ok(())
        },
    );
}

#[test]
fn with_unknown_option_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_reference(arc_process.clone()),
                unknown_option(arc_process.clone()),
            )
        },
        |(arc_process, reference, option)| {
            let options = arc_process.list_from_slice(&[option]).unwrap();

            prop_assert_badarg!(
                native(&arc_process, reference, options),
                "supported options are :flush or :info"
            );

            Ok(())
        },
    );
}

fn unknown_option(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    strategy::term(arc_process)
        .prop_filter("Option cannot be flush or info", |option| {
            match option.decode().unwrap() {
                TypedTerm::Atom(atom) => match atom.name() {
                    "flush" | "info" => false,
                    _ => true,
                },
                _ => true,
            }
        })
        .boxed()
}

fn r#type() -> Term {
    Atom::str_to_term("process")
}
