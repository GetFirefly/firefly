use super::*;

use std::thread;
use std::time::Duration;

#[test]
#[ignore]
fn without_timeout_returns_milliseconds_remaining_and_does_not_send_timeout_message() {
    with_timer(|milliseconds, message, timer_reference, process| {
        let half_milliseconds = milliseconds / 2;

        thread::sleep(Duration::from_millis(half_milliseconds + 1));
        timer::timeout();

        let timeout_message = timeout_message(timer_reference, message, process);

        // flaky
        assert!(!has_message(process, timeout_message));

        let first_milliseconds_remaining =
            erlang::read_timer_2(timer_reference, options(process), process)
                .expect("Timer could not be read");

        assert!(first_milliseconds_remaining.is_integer());
        // flaky
        assert!(process.integer(0).unwrap() < first_milliseconds_remaining);
        assert!(first_milliseconds_remaining <= process.integer(milliseconds / 2).unwrap());

        // again before timeout
        let second_milliseconds_remaining =
            erlang::read_timer_2(timer_reference, options(process), process)
                .expect("Timer could not be read");

        assert!(second_milliseconds_remaining.is_integer());
        assert!(second_milliseconds_remaining <= first_milliseconds_remaining);

        thread::sleep(Duration::from_millis(half_milliseconds + 1));
        timer::timeout();

        assert!(has_message(process, timeout_message));

        // again after timeout
        assert_eq!(
            erlang::read_timer_2(timer_reference, options(process), process),
            Ok(false.into())
        );
    })
}

#[test]
fn with_timeout_returns_false_after_timeout_message_was_sent() {
    with_timer(|milliseconds, message, timer_reference, process| {
        thread::sleep(Duration::from_millis(milliseconds + 1));
        timer::timeout();

        let timeout_message = timeout_message(timer_reference, message, process);

        assert!(
            has_message(process, timeout_message),
            "Mailbox contains: {:?}",
            process.mailbox.lock().borrow()
        );

        assert_eq!(
            erlang::read_timer_2(timer_reference, options(process), process),
            Ok(false.into())
        );

        // again
        assert_eq!(
            erlang::read_timer_2(timer_reference, options(process), process),
            Ok(false.into())
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
        same_thread_process_arc.integer(milliseconds).unwrap(),
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
