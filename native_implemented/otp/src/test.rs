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

use std::ptr::NonNull;
use std::str::FromStr;
use std::sync::Arc;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::{Atom, Pid, Term};
use firefly_rt::time::{Milliseconds, Monotonic};

use crate::erlang::{self, exit_1};
use crate::runtime::scheduler::{next_local_reference_term, Scheduled, SchedulerDependentAlloc};
use crate::runtime::time::monotonic;
use crate::runtime::timer;

pub fn exit_when_run(process: &Process, reason: Term) {
    todo!();
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

pub fn module() -> Result<Atom, AtomError> {
    Atom::from_str("test")
}

pub fn with_big_int(f: fn(&Process, Term) -> ()) {
    with_process(|process| {
        let big_int: Term = process.integer(Integer::MAX_SMALL + 1).unwrap();

        assert!(big_int.is_big_int());

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
    N: Fn(&Process, Term, Term) -> Result<Term, NonNull<ErlangException>>,
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
    result: fn(&Process, Term) -> Result<Term, NonNull<ErlangException>>,
) {
    with_options_with_timer_in_same_thread_with_timeout_returns_false_after_timeout_message_was_sent(
        |process, timer_reference, _| result(process, timer_reference),
        |_| Term::Nil,
    )
}

pub fn with_timer_in_same_thread<F>(f: F)
where
    F: FnOnce(Milliseconds, Term, Term, &Process) -> (),
{
    let same_thread_process_arc = process::default();
    let milliseconds = Milliseconds(100);

    let message: Term = Atom::str_to_term("message").into();
    let time: Term = next_local_reference_term.integer(milliseconds);
    let destination: Term = same_thread_process_arc.pid_term().unwrap();

    let timer_reference =
        erlang::start_timer_3::result(same_thread_process_arc.clone(), time, destination, message)
            .unwrap();

    f(
        milliseconds,
        message,
        timer_reference,
        &same_thread_process_arc,
    );
}

pub fn without_timer_returns_false(
    result: fn(&Process, Term) -> Result<Term, NonNull<ErlangException>>,
) {
    with_process(|process| {
        let timer_reference = next_local_reference_term(process).unwrap();

        assert_eq!(result(process, timer_reference), Ok(false.into()));
    });
}
