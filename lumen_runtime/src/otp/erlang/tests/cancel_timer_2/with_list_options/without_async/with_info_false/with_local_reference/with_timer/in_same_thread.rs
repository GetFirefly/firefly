use super::*;

use std::thread;
use std::time::Duration;

#[test]
fn without_timeout_returns_ok_and_does_not_send_timeout_message() {
    with_timer(|milliseconds, message, timer_reference, process| {
        let half_milliseconds = milliseconds / 2;

        thread::sleep(Duration::from_millis(half_milliseconds + 1));
        timer::timeout();

        let timeout_message = timeout_message(timer_reference, message, process);

        assert!(!has_message(process, timeout_message));

        assert_eq!(
            erlang::cancel_timer_2(timer_reference, options(process), process),
            Ok(Term::str_to_atom("ok", DoNotCare).unwrap())
        );

        // again before timeout
        assert_eq!(
            erlang::cancel_timer_2(timer_reference, options(process), process),
            Ok(Term::str_to_atom("ok", DoNotCare).unwrap())
        );

        thread::sleep(Duration::from_millis(half_milliseconds + 1));
        timer::timeout();

        assert!(!has_message(process, timeout_message));

        // again after timeout
        assert_eq!(
            erlang::cancel_timer_2(timer_reference, options(process), process),
            Ok(Term::str_to_atom("ok", DoNotCare).unwrap())
        );
    })
}

#[test]
fn with_timeout_returns_ok_after_timeout_message_was_sent() {
    with_timer(|milliseconds, message, timer_reference, process| {
        thread::sleep(Duration::from_millis(milliseconds + 1));
        timer::timeout();

        let timeout_message = timeout_message(timer_reference, message, process);

        assert!(
            has_message(process, timeout_message),
            "Mailbox contains: {:?}",
            process.mailbox.lock().unwrap()
        );

        assert_eq!(
            erlang::cancel_timer_2(timer_reference, options(process), process),
            Ok(Term::str_to_atom("ok", DoNotCare).unwrap())
        );

        // again
        assert_eq!(
            erlang::cancel_timer_2(timer_reference, options(process), process),
            Ok(Term::str_to_atom("ok", DoNotCare).unwrap())
        );
    })
}

fn with_timer<F>(f: F)
where
    F: FnOnce(u64, Term, Term, &Process) -> (),
{
    let same_thread_process_arc = process::local::new();
    let milliseconds: u64 = 100;

    let message = Term::str_to_atom("message", DoNotCare).unwrap();
    let timer_reference = erlang::start_timer_3(
        milliseconds.into_process(&same_thread_process_arc),
        same_thread_process_arc.pid,
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
