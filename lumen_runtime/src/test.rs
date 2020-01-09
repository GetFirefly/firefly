pub mod r#loop;
pub mod process_dictionary;

// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest,
// so disable property-based tests and associated helpers completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
pub mod proptest;
#[cfg(all(not(target_arch = "wasm32"), test))]
pub mod strategy;

#[cfg(all(not(target_arch = "wasm32"), test))]
pub use self::proptest::*;

use std::convert::TryInto;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::{Process, Status};
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang;
use crate::process::{self, SchedulerDependentAlloc};
use crate::scheduler::with_process;
use crate::time::Milliseconds;
use crate::timer;

pub fn assert_exits<F: Fn(Option<Term>)>(
    process: &Process,
    expected_reason: Term,
    assert_stacktrace: F,
    source_substring: &str,
) {
    match *process.status.read() {
        Status::Exiting(ref runtime_exception) => {
            assert_eq!(runtime_exception.reason(), Some(expected_reason));
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
            let stacktrace_boxed_cons: Boxed<Cons> = stacktrace.unwrap().try_into().unwrap();
            let head = stacktrace_boxed_cons.head;

            assert_eq!(
                head,
                process
                    .tuple_from_slice(&[module, function, arguments, Term::NIL])
                    .unwrap()
            );
        },
        source_substring,
    );
}

pub fn badarity_reason(process: &Process, function: Term, args: Term) -> Term {
    let tag = atom!("badarity");
    let fun_args = process.tuple_from_slice(&[function, args]).unwrap();

    process.tuple_from_slice(&[tag, fun_args]).unwrap()
}

pub fn timeout_after_half(milliseconds: Milliseconds) {
    thread::sleep(Duration::from_millis(milliseconds / 2 + 1));
    timer::timeout();
}

pub fn timeout_after_half_and_wait(milliseconds: Milliseconds, barrier: &Barrier) {
    timeout_after_half(milliseconds);
    barrier.wait();
}

pub fn wait_for_completion(barrier: &Barrier) {
    barrier.wait();
}

pub fn wait_for_message(barrier: &Barrier) {
    barrier.wait();
}

pub fn with_big_int(f: fn(&Process, Term) -> ()) {
    with_process(|process| {
        let big_int: Term = process.integer(SmallInteger::MAX_VALUE + 1).unwrap();

        assert!(big_int.is_boxed_bigint());

        f(&process, big_int)
    })
}

pub fn with_timeout_returns_false_after_timeout_message_was_sent(
    native: fn(&Process, Term) -> exception::Result<Term>,
) {
    with_timer_in_same_thread(|milliseconds, message, timer_reference, process| {
        thread::sleep(Duration::from_millis(milliseconds + 1));
        timer::timeout();

        let timeout_message = timeout_message(timer_reference, message, process);

        assert_has_message!(process, timeout_message);

        assert_eq!(native(process, timer_reference), Ok(false.into()));

        // again
        assert_eq!(native(process, timer_reference), Ok(false.into()));
    })
}

pub fn with_timer_in_different_thread<F>(f: F)
where
    F: FnOnce(u64, &Barrier, Term, &Process) -> (),
{
    let same_thread_process_arc = process::test(&process::test_init());
    let milliseconds: u64 = 100;

    // no wait to receive implemented yet, so use barrier for signalling
    let same_thread_barrier = Arc::new(Barrier::new(2));

    let different_thread_same_thread_process_arc = Arc::clone(&same_thread_process_arc);
    let different_thread_barrier = same_thread_barrier.clone();

    let different_thread = thread::spawn(move || {
        let different_thread_process_arc = process::test(&different_thread_same_thread_process_arc);
        let same_thread_pid = different_thread_same_thread_process_arc.pid();

        let timer_reference = erlang::start_timer_3::native(
            different_thread_process_arc.clone(),
            different_thread_process_arc.integer(milliseconds).unwrap(),
            same_thread_pid.into(),
            Atom::str_to_term("different"),
        )
        .unwrap();

        erlang::send_2::native(
            &different_thread_process_arc,
            same_thread_pid.into(),
            different_thread_process_arc
                .tuple_from_slice(&[Atom::str_to_term("timer_reference"), timer_reference])
                .unwrap(),
        )
        .expect("Different thread could not send to same thread");

        wait_for_message(&different_thread_barrier);
        timeout_after_half_and_wait(milliseconds, &different_thread_barrier);
        timeout_after_half_and_wait(milliseconds, &different_thread_barrier);

        // stops Drop of scheduler ID
        wait_for_completion(&different_thread_barrier);
    });

    wait_for_message(&same_thread_barrier);

    let timer_reference_tuple =
        receive_message(&same_thread_process_arc).expect("Cross-thread receive failed");

    let timer_reference = erlang::element_2::native(
        same_thread_process_arc.integer(2).unwrap(),
        timer_reference_tuple,
    )
    .unwrap();

    f(
        milliseconds,
        &same_thread_barrier,
        timer_reference,
        &same_thread_process_arc,
    );

    wait_for_completion(&same_thread_barrier);

    different_thread
        .join()
        .expect("Could not join different thread");
}

pub fn with_timer_in_same_thread<F>(f: F)
where
    F: FnOnce(u64, Term, Term, &Process) -> (),
{
    let same_thread_process_arc = process::test(&process::test_init());
    let milliseconds: u64 = 100;

    let message = Atom::str_to_term("message");
    let timer_reference = erlang::start_timer_3::native(
        same_thread_process_arc.clone(),
        same_thread_process_arc.integer(milliseconds).unwrap(),
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

pub fn without_timer_returns_false(native: fn(&Process, Term) -> exception::Result<Term>) {
    with_process(|process| {
        let timer_reference = process.next_reference().unwrap();

        assert_eq!(native(process, timer_reference), Ok(false.into()));
    });
}
