use super::*;

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

#[test]
fn without_timeout_returns_milliseconds_remaining() {
    with_timer(|milliseconds, barrier, timer_reference, process| {
        timeout_after_half(milliseconds, barrier);

        let message = atom_unchecked("different");
        let timeout_message = timeout_message(timer_reference, message, process);

        assert!(!has_message(process, timeout_message));

        assert_eq!(
            erlang::read_timer_2(timer_reference, options(process), process),
            Ok(atom_unchecked("ok"))
        );

        let first_received_message = receive_message(process).unwrap();

        let first_received_tuple_result: core::result::Result<Boxed<Tuple>, _> =
            first_received_message.try_into();

        assert!(first_received_tuple_result.is_ok());

        let first_received_tuple = first_received_tuple_result.unwrap();

        assert_eq!(first_received_tuple[0], atom_unchecked("read_timer"));
        assert_eq!(first_received_tuple[1], timer_reference);

        let first_milliseconds_remaining = first_received_tuple[2];

        assert!(first_milliseconds_remaining.is_integer());
        assert!(process.integer(0).unwrap() < first_milliseconds_remaining);
        assert!(first_milliseconds_remaining <= process.integer(milliseconds / 2).unwrap());

        // again before timeout
        assert_eq!(
            erlang::read_timer_2(timer_reference, options(process), process),
            Ok(atom_unchecked("ok"))
        );

        let second_received_message = receive_message(process).unwrap();

        let second_received_tuple_result: core::result::Result<Boxed<Tuple>, _> =
            second_received_message.try_into();

        assert!(second_received_tuple_result.is_ok());

        let second_received_tuple = second_received_tuple_result.unwrap();

        assert_eq!(second_received_tuple[0], atom_unchecked("read_timer"));
        assert_eq!(second_received_tuple[1], timer_reference);

        let second_milliseconds_remaining = second_received_tuple[2];

        assert!(second_milliseconds_remaining.is_integer());
        assert!(second_milliseconds_remaining <= first_milliseconds_remaining);

        timeout_after_half(milliseconds, barrier);

        assert_eq!(receive_message(process), Some(timeout_message));

        let false_read_timer_message = read_timer_message(timer_reference, false.into(), process);

        // again after timeout
        assert_eq!(
            erlang::read_timer_2(timer_reference, options(process), process),
            Ok(atom_unchecked("ok"))
        );
        assert_eq!(receive_message(process), Some(false_read_timer_message))
    });
}

#[test]
fn with_timeout_returns_ok_after_timeout_message_was_sent() {
    with_timer(|milliseconds, barrier, timer_reference, process| {
        timeout_after_half(milliseconds, barrier);
        timeout_after_half(milliseconds, barrier);

        let message = atom_unchecked("different");
        let timeout_message = timeout_message(timer_reference, message, process);

        assert_eq!(receive_message(process), Some(timeout_message));

        let read_timer_message = read_timer_message(timer_reference, false.into(), process);

        assert_eq!(
            erlang::read_timer_2(timer_reference, options(process), process),
            Ok(atom_unchecked("ok"))
        );
        assert_eq!(receive_message(process), Some(read_timer_message));

        // again
        assert_eq!(
            erlang::read_timer_2(timer_reference, options(process), process),
            Ok(atom_unchecked("ok"))
        );
        assert_eq!(receive_message(process), Some(read_timer_message));
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
