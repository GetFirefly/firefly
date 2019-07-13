use super::*;

use std::thread;
use std::time::Duration;

#[test]
fn without_timeout_returns_milliseconds_remaining_and_does_not_send_timeout_message() {
    with_timer(|milliseconds, message, timer_reference, process| {
        let half_milliseconds = milliseconds / 2;

        thread::sleep(Duration::from_millis(half_milliseconds + 1));
        timer::timeout().unwrap();

        let timeout_message = timeout_message(timer_reference, message, process);

        assert!(!has_message(process, timeout_message));

        assert_eq!(
            erlang::cancel_timer_2(timer_reference, options(process), process),
            Ok(atom_unchecked("ok"))
        );

        // again before timeout
        assert_eq!(
            erlang::cancel_timer_2(timer_reference, options(process), process),
            Ok(atom_unchecked("ok"))
        );

        thread::sleep(Duration::from_millis(half_milliseconds + 1));
        timer::timeout().unwrap();

        assert!(!has_message(process, timeout_message));

        // again after timeout
        assert_eq!(
            erlang::cancel_timer_2(timer_reference, options(process), process),
            Ok(atom_unchecked("ok"))
        );
    })
}

#[test]
fn with_timeout_returns_false_after_timeout_message_was_sent() {
    with_timer(|milliseconds, message, timer_reference, process| {
        thread::sleep(Duration::from_millis(milliseconds + 1));
        timer::timeout().unwrap();

        let timeout_message = timeout_message(timer_reference, message, process);

        assert!(
            has_message(process, timeout_message),
            "Mailbox contains: {:?}",
            process.mailbox.lock().borrow()
        );

        assert_eq!(
            erlang::cancel_timer_2(timer_reference, options(process), process),
            Ok(atom_unchecked("ok"))
        );

        // again
        assert_eq!(
            erlang::cancel_timer_2(timer_reference, options(process), process),
            Ok(atom_unchecked("ok"))
        );
    })
}

fn with_timer<F>(f: F)
where
    F: FnOnce(u64, Term, Term, &ProcessControlBlock) -> (),
{
    let same_thread_process_arc = process::test(&process::test_init());
    let milliseconds: u64 = 100;

    let message = atom_unchecked("message");
    let timer_reference = erlang::start_timer_3(
        same_thread_process_arc.integer(milliseconds),
        unsafe { same_thread_process_arc.pid().as_term() },
        message,
        same_thread_process_arc.clone(),
    )
    .unwrap();

    f(
        milliseconds,
        message,
        timer_reference,
        &same_thread_process_arc,
    );
}
