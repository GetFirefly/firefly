pub mod anonymous_0;
pub mod anonymous_1;
mod init;
pub mod loop_0;
pub mod process;
pub mod return_from_fn_0;
pub mod return_from_fn_1;

// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest,
// so disable property-based tests and associated helpers completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod proptest;
#[cfg(all(not(target_arch = "wasm32"), test))]
pub mod strategy;

#[cfg(all(not(target_arch = "wasm32"), test))]
pub use self::proptest::*;

use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::trace::Trace;
use liblumen_alloc::erts::process::{Process, Status};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::time::{Milliseconds, Monotonic};

use crate::erlang::{self, exit_1};
use crate::runtime::scheduler::{Scheduled, SchedulerDependentAlloc};
use crate::runtime::time::monotonic;
use crate::runtime::timer;

pub fn assert_exits<F: Fn(Option<Arc<Trace>>)>(
    process: &Process,
    expected_reason: Term,
    assert_stacktrace: F,
    source_substring: &str,
) {
    match *process.status.read() {
        Status::RuntimeException(ref runtime_exception) => {
            assert_eq!(runtime_exception.reason(), expected_reason);
            assert_stacktrace(runtime_exception.stacktrace());

            let source_string = format!("{:?}", runtime_exception.source());

            assert!(
                source_string.contains(source_substring),
                "source ({}) does not contain `{}`",
                source_string,
                source_substring
            );
        }
        ref status => panic!("Process status ({:?}) is not exiting.", status),
    };
}

pub fn assert_exits_badarith(process: &Process, source_substring: &str) {
    assert_exits(process, atom!("badarith"), |_| {}, source_substring)
}

pub fn assert_exits_undef(
    process: &Process,
    module: Term,
    function: Term,
    arguments: Term,
    source_substring: &str,
) {
    assert_exits(
        process,
        atom!("undef"),
        |stacktrace| {
            let stacktrace_term = stacktrace.unwrap().as_term().unwrap();
            let stacktrace_boxed_cons: Boxed<Cons> = stacktrace_term.try_into().unwrap();
            let head = stacktrace_boxed_cons.head;

            assert_eq!(
                head,
                process.tuple_from_slice(&[module, function, arguments, Term::NIL])
            );
        },
        source_substring,
    );
}

pub fn exit_when_run(process: &Process, reason: Term) {
    process.queue_frame_with_arguments(exit_1::frame().with_arguments(false, &[reason]));
    process.stack_queued_frames_with_arguments();
    process.scheduler().unwrap().stop_waiting(process);
}

pub fn freeze_timeout() -> Monotonic {
    let frozen = monotonic::freeze();
    timer::timeout();

    frozen
}

pub fn freeze_at_timeout(frozen: Monotonic) {
    monotonic::freeze_at(frozen);
    timer::timeout();
}

pub fn module() -> Atom {
    Atom::from_str("test")
}

pub fn module_id() -> usize {
    module().id()
}

pub fn with_big_int(f: fn(&Process, Term) -> ()) {
    with_process(|process| {
        let big_int: Term = process.integer(SmallInteger::MAX_VALUE + 1);

        assert!(big_int.is_boxed_bigint());

        f(&process, big_int)
    })
}

pub fn with_process<F>(f: F)
where
    F: FnOnce(&Process) -> (),
{
    f(&process::default())
}

pub fn with_process_arc<F>(f: F)
where
    F: FnOnce(Arc<Process>) -> (),
{
    f(process::default())
}

pub fn with_options_with_timer_in_same_thread_with_timeout_returns_false_after_timeout_message_was_sent<
    N,
    O,
>(
    result: N,
    options: O,
) where
    N: Fn(&Process, Term, Term) -> exception::Result<Term>,
    O: Fn(&Process) -> Term,
{
    with_timer_in_same_thread(|milliseconds, message, timer_reference, process| {
        let start_time_in_milliseconds = freeze_timeout();
        freeze_at_timeout(start_time_in_milliseconds + milliseconds + Milliseconds(1));

        let timeout_message = timeout_message(timer_reference, message, process);

        assert_has_message!(process, timeout_message);

        assert_eq!(
            result(process, timer_reference, options(process)),
            Ok(false.into())
        );
        // again
        assert_eq!(
            result(process, timer_reference, options(process)),
            Ok(false.into())
        );
    })
}

pub fn with_timer_in_same_thread_with_timeout_returns_false_after_timeout_message_was_sent(
    result: fn(&Process, Term) -> exception::Result<Term>,
) {
    with_options_with_timer_in_same_thread_with_timeout_returns_false_after_timeout_message_was_sent(
        |process, timer_reference, _| result(process, timer_reference),
        |_| Term::NIL,
    )
}

pub fn with_timer_in_same_thread<F>(f: F)
where
    F: FnOnce(Milliseconds, Term, Term, &Process) -> (),
{
    let same_thread_process_arc = process::default();
    let milliseconds = Milliseconds(100);

    let message = Atom::str_to_term("message");
    let timer_reference = erlang::start_timer_3::result(
        same_thread_process_arc.clone(),
        same_thread_process_arc.integer(milliseconds),
        same_thread_process_arc.pid().into(),
        message,
    )
    .unwrap();

    f(
        milliseconds,
        message,
        timer_reference,
        &same_thread_process_arc,
    );
}

pub fn without_timer_returns_false(result: fn(&Process, Term) -> exception::Result<Term>) {
    with_process(|process| {
        let timer_reference = process.next_reference();

        assert_eq!(result(process, timer_reference), Ok(false.into()));
    });
}
