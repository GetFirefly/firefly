use super::*;

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

#[test]
fn without_timeout_returns_milliseconds_remaining_and_does_not_send_timeout_message() {
    with_timer(|milliseconds, barrier, timer_reference, process| {
        timeout_after_half(milliseconds, barrier);

        let message = Term::str_to_atom("different", DoNotCare).unwrap();
        let timeout_message = timeout_message(timer_reference, message, process);

        assert!(!has_message(process, timeout_message));

        assert_eq!(
            erlang::cancel_timer_2(timer_reference, options(process), process),
            Ok(Term::str_to_atom("ok", DoNotCare).unwrap())
        );

        let received_message = receive_message(process).unwrap();

        assert_eq!(received_message.tag(), Boxed);

        let unboxed_received_message: &Term = received_message.unbox_reference();

        assert_eq!(unboxed_received_message.tag(), Arity);

        let received_tuple: &Tuple = received_message.unbox_reference();

        assert_eq!(
            received_tuple[0],
            Term::str_to_atom("cancel_timer", DoNotCare).unwrap()
        );
        assert_eq!(received_tuple[1], timer_reference);

        let milliseconds_remaining = received_tuple[2];

        assert!(milliseconds_remaining.is_integer());
        assert!(0.into_process(process) < milliseconds_remaining);
        assert!(milliseconds_remaining <= (milliseconds / 2).into_process(process));

        let false_cancel_timer_message =
            cancel_timer_message(timer_reference, false.into(), process);

        // again before timeout
        assert_eq!(
            erlang::cancel_timer_2(timer_reference, options(process), process),
            Ok(Term::str_to_atom("ok", DoNotCare).unwrap())
        );
        assert_eq!(receive_message(process), Some(false_cancel_timer_message));

        timeout_after_half(milliseconds, barrier);

        assert!(!has_message(process, timeout_message));

        // again after timeout
        assert_eq!(
            erlang::cancel_timer_2(timer_reference, options(process), process),
            Ok(Term::str_to_atom("ok", DoNotCare).unwrap())
        );
        assert_eq!(receive_message(process), Some(false_cancel_timer_message));
    });
}

#[test]
fn with_timeout_returns_false_after_timeout_message_was_sent() {
    with_timer(|milliseconds, barrier, timer_reference, process| {
        timeout_after_half(milliseconds, barrier);
        timeout_after_half(milliseconds, barrier);

        let message = Term::str_to_atom("different", DoNotCare).unwrap();
        let timeout_message = timeout_message(timer_reference, message, process);

        assert_eq!(receive_message(process), Some(timeout_message));

        let cancel_timer_message = cancel_timer_message(timer_reference, false.into(), process);

        assert_eq!(
            erlang::cancel_timer_2(timer_reference, options(process), process),
            Ok(Term::str_to_atom("ok", DoNotCare).unwrap())
        );
        assert_eq!(receive_message(process), Some(cancel_timer_message));

        // again
        assert_eq!(
            erlang::cancel_timer_2(timer_reference, options(process), process),
            Ok(Term::str_to_atom("ok", DoNotCare).unwrap())
        );
        assert_eq!(receive_message(process), Some(cancel_timer_message));
    });
}

fn with_timer<F>(f: F)
where
    F: FnOnce(u64, &Barrier, Term, &Process) -> (),
{
    let same_thread_process_arc = process::local::new();
    let milliseconds: u64 = 100;

    // no wait to receive implemented yet, so use barrier for signalling
    let same_thread_barrier = Arc::new(Barrier::new(2));

    let different_thread_same_thread_process_pid = same_thread_process_arc.pid.clone();
    let different_thread_barrier = same_thread_barrier.clone();

    let different_thread = thread::spawn(move || {
        let different_thread_process_arc = process::local::new();

        let timer_reference = erlang::start_timer_3(
            milliseconds.into_process(&different_thread_process_arc),
            different_thread_same_thread_process_pid,
            Term::str_to_atom("different", DoNotCare).unwrap(),
            different_thread_process_arc.clone(),
        )
        .unwrap();

        erlang::send_2(
            different_thread_same_thread_process_pid,
            Term::slice_to_tuple(
                &[
                    Term::str_to_atom("timer_reference", DoNotCare).unwrap(),
                    timer_reference,
                ],
                &different_thread_process_arc,
            ),
            &different_thread_process_arc,
        )
        .expect("Different thread could not send to same thread");

        wait_for_message(&different_thread_barrier);
        timeout_after_half(milliseconds, &different_thread_barrier);
        timeout_after_half(milliseconds, &different_thread_barrier);

        // stops Drop of scheduler ID
        wait_for_completion(&different_thread_barrier);
    });

    wait_for_message(&same_thread_barrier);

    let timer_reference_tuple =
        receive_message(&same_thread_process_arc).expect("Cross-thread receive failed");

    let timer_reference = erlang::element_2(
        timer_reference_tuple,
        2.into_process(&same_thread_process_arc),
        &same_thread_process_arc,
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

fn timeout_after_half(milliseconds: Milliseconds, barrier: &Barrier) {
    thread::sleep(Duration::from_millis(milliseconds / 2 + 1));
    timer::timeout();
    barrier.wait();
}

fn wait_for_completion(barrier: &Barrier) {
    barrier.wait();
}

fn wait_for_message(barrier: &Barrier) {
    barrier.wait();
}
