mod with_process_type;

use std::sync::Arc;

use proptest::prop_assert_eq;
use proptest::strategy::{BoxedStrategy, Strategy};
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{Term, TypedTerm};

use crate::otp::erlang::monitor_2::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_supported_type_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&unsupported_type(arc_process.clone()), |r#type| {
                prop_assert_eq!(
                    native(&arc_process, r#type, arc_process.pid_term()),
                    Err(badarg!().into())
                );

                Ok(())
            })
            .unwrap();
    });
}

fn unsupported_type(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    strategy::term(arc_process)
        .prop_filter("Type cannot be :process", |r#type| {
            match r#type.to_typed_term().unwrap() {
                TypedTerm::Atom(atom) => match atom.name() {
                    "process" | "port" | "time_offset" => false,
                    _ => true,
                },
                _ => true,
            }
        })
        .boxed()
}
