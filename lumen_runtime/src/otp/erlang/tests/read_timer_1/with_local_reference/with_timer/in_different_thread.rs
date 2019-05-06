use super::*;

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

#[test]
fn without_timeout_returns_milliseconds_remaining() {
    with_timer(|milliseconds, barrier, timer_reference, process| {
        timeout_after_half(milliseconds, barrier);

        let message = Term::str_to_atom("different", DoNotCare).unwrap();
        let timeout_message = timeout_message(timer_reference, message, process);

        assert!(!has_message(process, timeout_message));

        let first_milliseconds_remaining =
            erlang::read_timer_1(timer_reference, process).expect("Timer could not be read");

        assert!(first_milliseconds_remaining.is_integer());
        assert!(0.into_process(process) < first_milliseconds_remaining);
        assert!(first_milliseconds_remaining <= (milliseconds / 2).into_process(process));

        // again before timeout
        let second_milliseconds_remaining =
            erlang::read_timer_1(timer_reference, process).expect("Timer could not be read");

        assert!(second_milliseconds_remaining.is_integer());
        assert!(second_milliseconds_remaining <= first_milliseconds_remaining);

        timeout_after_half(milliseconds, barrier);

        assert!(has_message(process, timeout_message));

        // again after timeout
        assert_eq!(
            erlang::read_timer_1(timer_reference, process),
            Ok(false.into())
        );
    });
}

#[test]
fn with_timeout_returns_false_after_timeout_message_was_sent() {
    with_timer(|milliseconds, barrier, timer_reference, process| {
        timeout_after_half(milliseconds, barrier);
        timeout_after_half(milliseconds, barrier);

        let message = Term::str_to_atom("different", DoNotCare).unwrap();
        let timeout_message = timeout_message(timer_reference, message, process);

        assert!(
            has_message(process, timeout_message),
            "Mailbox does not contain {:?} and instead contains {:?}",
            timeout_message,
            process.mailbox.lock().unwrap()
        );

        assert_eq!(
            erlang::read_timer_1(timer_reference, process),
            Ok(false.into())
        );

        // again
        assert_eq!(
            erlang::read_timer_1(timer_reference, process),
            Ok(false.into())
        );
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
