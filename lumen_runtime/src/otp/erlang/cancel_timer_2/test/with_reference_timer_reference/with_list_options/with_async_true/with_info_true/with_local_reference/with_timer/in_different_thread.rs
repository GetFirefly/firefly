use super::*;

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

#[test]
#[ignore]
fn without_timeout_returns_milliseconds_remaining_and_does_not_send_timeout_message() {
    with_timer(|milliseconds, barrier, timer_reference, process| {
        timeout_after_half(milliseconds, barrier);

        let message = Atom::str_to_term("different");
        let timeout_message = timeout_message(timer_reference, message, process);

        // flaky
        assert!(!has_message(process, timeout_message));

        assert_eq!(
            native(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );

        let received_message = receive_message(process).unwrap();

        let received_tuple_result: core::result::Result<Boxed<Tuple>, _> =
            received_message.try_into();

        assert!(received_tuple_result.is_ok());

        let received_tuple = received_tuple_result.unwrap();

        assert_eq!(received_tuple[0], Atom::str_to_term("cancel_timer"));
        assert_eq!(received_tuple[1], timer_reference);

        let milliseconds_remaining = received_tuple[2];

        assert!(milliseconds_remaining.is_integer());
        assert!(process.integer(0).unwrap() < milliseconds_remaining);
        assert!(milliseconds_remaining <= process.integer(milliseconds / 2).unwrap());

        let false_cancel_timer_message =
            cancel_timer_message(timer_reference, false.into(), process);

        // again before timeout
        assert_eq!(
            native(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );
        assert_eq!(receive_message(process), Some(false_cancel_timer_message));

        timeout_after_half(milliseconds, barrier);

        assert!(!has_message(process, timeout_message));

        // again after timeout
        assert_eq!(
            native(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );
        assert_eq!(receive_message(process), Some(false_cancel_timer_message));
    });
}

#[test]
fn with_timeout_returns_false_after_timeout_message_was_sent() {
    with_timer(|milliseconds, barrier, timer_reference, process| {
        timeout_after_half(milliseconds, barrier);
        timeout_after_half(milliseconds, barrier);

        let message = Atom::str_to_term("different");
        let timeout_message = timeout_message(timer_reference, message, process);

        assert_eq!(receive_message(process), Some(timeout_message));

        let cancel_timer_message = cancel_timer_message(timer_reference, false.into(), process);

        assert_eq!(
            native(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );
        assert_eq!(receive_message(process), Some(cancel_timer_message));

        // again
        assert_eq!(
            native(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );
        assert_eq!(receive_message(process), Some(cancel_timer_message));
    });
}

fn with_timer<F>(f: F)
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
        timeout_after_half(milliseconds, &different_thread_barrier);
        timeout_after_half(milliseconds, &different_thread_barrier);

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
