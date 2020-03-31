mod with_reference;

use std::convert::TryInto;
use std::sync::Arc;

use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Just, Strategy};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::read_timer_2::native;
use crate::runtime::scheduler::SchedulerDependentAlloc;
use crate::runtime::time::monotonic;
use crate::test::{
    external_arc_node, freeze_at_timeout, freeze_timeout, has_message, receive_message, strategy,
    timeout_message, timer_message, with_process, with_timer_in_same_thread,
};

#[test]
fn without_reference_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_reference(arc_process.clone()),
                options(arc_process.clone()),
            )
        },
        |(arc_process, timer_reference, options)| {
            prop_assert_badarg!(
                native(&arc_process, timer_reference, options),
                format!(
                    "timer_reference ({}) is not a local reference",
                    timer_reference
                )
            );

            Ok(())
        },
    );
}

#[test]
fn with_reference_without_list_options_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_reference(arc_process.clone()),
                strategy::term::is_not_list(arc_process.clone()),
            )
        },
        |(arc_process, timer_reference, options)| {
            prop_assert_badarg!(
                native(&arc_process, timer_reference, options),
                "improper list"
            );

            Ok(())
        },
    );
}

fn async_option(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    strategy::term::is_boolean()
        .prop_map(move |async_value| {
            arc_process
                .tuple_from_slice(&[Atom::str_to_term("async"), async_value])
                .unwrap()
        })
        .boxed()
}

fn options(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![
        Just(Term::NIL),
        async_option(arc_process.clone()).prop_map(move |async_option| {
            arc_process.list_from_slice(&[async_option]).unwrap()
        })
    ]
    .boxed()
}

fn read_timer_message(timer_reference: Term, result: Term, process: &Process) -> Term {
    timer_message("read_timer", timer_reference, result, process)
}
