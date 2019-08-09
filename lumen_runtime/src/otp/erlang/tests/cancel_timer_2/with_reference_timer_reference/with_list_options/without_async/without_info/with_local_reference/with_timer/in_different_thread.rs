use super::*;

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

#[test]
fn without_timeout_returns_milliseconds_remaining_and_does_not_send_timeout_message() {
    with_timer(|milliseconds, barrier, timer_reference, process| {
        timeout_after_half(milliseconds, barrier);

        let message = atom_unchecked("different");
        let timeout_message = timeout_message(timer_reference, message, process);

        assert!(!has_message(process, timeout_message));

        let milliseconds_remaining =
            erlang::cancel_timer_2(timer_reference, options(process), process)
                .expect("Timer could not be cancelled");

        assert!(milliseconds_remaining.is_integer());
        assert!(process.integer(0).unwrap() < milliseconds_remaining);
        assert!(milliseconds_remaining <= process.integer(milliseconds / 2).unwrap());

        // again before timeout
        assert_eq!(
            erlang::cancel_timer_2(timer_reference, options(process), process),
            Ok(false.into())
        );

        timeout_after_half(milliseconds, barrier);

        assert!(!has_message(process, timeout_message));

        // again after timeout
        assert_eq!(
            erlang::cancel_timer_2(timer_reference, options(process), process),
            Ok(false.into())
        );
    });
}

#[test]
fn with_timeout_returns_false_after_timeout_message_was_sent() {
    with_timer(|milliseconds, barrier, timer_reference, process| {
        timeout_after_half(milliseconds, barrier);
        timeout_after_half(milliseconds, barrier);

        let message = atom_unchecked("different");
        let timeout_message = timeout_message(timer_reference, message, process);

        assert!(
            has_message(process, timeout_message),
            "Mailbox does not contain {:?} and instead contains {:?}",
            timeout_message,
            process.mailbox.lock().borrow()
        );

        assert_eq!(
            erlang::cancel_timer_2(timer_reference, options(process), process),
            Ok(false.into())
        );

        // again
        assert_eq!(
            erlang::cancel_timer_2(timer_reference, options(process), process),
            Ok(false.into())
        );
    });
}

fn with_timer<F>(f: F)
where
    F: FnOnce(u64, &Barrier, Term, &ProcessControlBlock) -> (),
{
    let same_thread_process_arc = process::test(&process::test_init());
    let milliseconds: u64 = 100;

    // no wait to receive implemented yet, so use barrier for signalling
    let same_thread_barrier = Arc::new(Barrier::new(2));

    let different_thread_same_thread_process_arc = Arc::clone(&same_thread_process_arc);
    let different_thread_barrier = same_thread_barrier.clone();

    let different_thread = thread::spawn(move || {
        let different_thread_process_arc = process::test(&different_thread_same_thread_process_arc);
        let same_thread_pid = unsafe { different_thread_same_thread_process_arc.pid().as_term() };

        let timer_reference = erlang::start_timer_3(
            different_thread_process_arc.integer(milliseconds).unwrap(),
            same_thread_pid,
            atom_unchecked("different"),
            different_thread_process_arc.clone(),
        )
        .unwrap();

        erlang::send_2(
            same_thread_pid,
            different_thread_process_arc
                .tuple_from_slice(&[atom_unchecked("timer_reference"), timer_reference])
                .unwrap(),
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
    timer::timeout().unwrap();
    barrier.wait();
}

fn wait_for_completion(barrier: &Barrier) {
    barrier.wait();
}

fn wait_for_message(barrier: &Barrier) {
    barrier.wait();
}
