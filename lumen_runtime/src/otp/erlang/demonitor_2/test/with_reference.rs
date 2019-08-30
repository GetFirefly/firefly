mod with_flush_and_info_options;
mod with_flush_option;
mod with_info_option;
mod without_options;

use super::*;

use std::sync::Arc;

use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::term::atom_unchecked;

use crate::otp::erlang::exit_1;
use crate::process;
use crate::scheduler::Scheduler;
use crate::test::{has_message, monitor_count, monitored_count};

#[test]
fn without_proper_list_for_options_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_reference(arc_process.clone()),
                    strategy::term::is_not_proper_list(arc_process.clone()),
                ),
                |(reference, options)| {
                    prop_assert_eq!(
                        native(&arc_process, reference, options),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_unknown_option_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_reference(arc_process.clone()),
                    unknown_option(arc_process.clone()),
                ),
                |(reference, option)| {
                    let options = arc_process.list_from_slice(&[option]).unwrap();

                    prop_assert_eq!(
                        native(&arc_process, reference, options),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn unknown_option(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    strategy::term(arc_process)
        .prop_filter("Option cannot be flush or info", |option| {
            match option.to_typed_term().unwrap() {
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
    atom_unchecked("process")
}
