mod with_process_type;

use std::sync::Arc;

use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Just, Strategy};
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::monitor_2::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_supported_type_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                unsupported_type(arc_process.clone()),
            )
        },
        |(arc_process, r#type)| {
            prop_assert_badarg!(
                native(&arc_process, r#type, arc_process.pid_term()),
                "supported types are :port, :process, or :time_offset"
            );

            Ok(())
        },
    );
}

fn unsupported_type(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    strategy::term(arc_process)
        .prop_filter("Type cannot be :process", |r#type| {
            match r#type.decode().unwrap() {
                TypedTerm::Atom(atom) => match atom.name() {
                    "process" | "port" | "time_offset" => false,
                    _ => true,
                },
                _ => true,
            }
        })
        .boxed()
}
