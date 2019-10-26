mod with_registered_name;

use super::*;

use std::sync::Arc;

use proptest::strategy::{BoxedStrategy, Strategy};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::{Term, TypedTerm, Atom, Pid};

#[test]
fn without_supported_item_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&unsupported_item(arc_process.clone()), |item| {
                let pid = arc_process.pid_term();
                prop_assert_eq!(native(&arc_process, pid, item), Err(badarg!().into()));

                Ok(())
            })
            .unwrap();
    });
}

fn unsupported_item(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    strategy::term(arc_process)
        .prop_filter("Item cannot be supported", |item| {
            match item.decode().unwrap() {
                TypedTerm::Atom(atom) => match atom.name() {
                    "registered_name" => false,
                    _ => true,
                },
                _ => true,
            }
        })
        .boxed()
}
