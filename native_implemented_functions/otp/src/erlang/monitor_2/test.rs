mod with_process_type;

use std::convert::TryInto;
use std::sync::Arc;

use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Just, Strategy};
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::{registry, scheduler};

use crate::erlang::monitor_2::result;
use crate::erlang::node_0;
use crate::test::{self, *};

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
                result(&arc_process, r#type, arc_process.pid_term()),
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
