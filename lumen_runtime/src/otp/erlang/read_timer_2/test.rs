mod with_reference;

use std::convert::TryInto;
use std::sync::Arc;

use proptest::prop_assert_eq;
use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Just, Strategy};
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{
    atom_unchecked, make_pid, AsTerm, Boxed, SmallInteger, Term, Tuple,
};

use crate::otp::erlang;
use crate::otp::erlang::read_timer_2::native;
use crate::process;
use crate::process::SchedulerDependentAlloc;
use crate::scheduler::{with_process, with_process_arc};
use crate::test::strategy;
use crate::test::{has_message, receive_message, timeout_message, timer_message};
use crate::time::Milliseconds;
use crate::timer;

#[test]
fn without_reference_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_reference(arc_process.clone()),
                    options(arc_process.clone()),
                ),
                |(timer_reference, options)| {
                    prop_assert_eq!(
                        native(&arc_process, timer_reference, options),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_reference_without_list_options_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_reference(arc_process.clone()),
                    strategy::term::is_not_list(arc_process.clone()),
                ),
                |(timer_reference, options)| {
                    prop_assert_eq!(
                        native(&arc_process, timer_reference, options),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn async_option(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    strategy::term::is_boolean()
        .prop_map(move |async_value| {
            arc_process
                .tuple_from_slice(&[atom_unchecked("async"), async_value])
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
