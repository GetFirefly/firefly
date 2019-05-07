use super::*;

use std::thread;
use std::time::Duration;

#[test]
fn without_timeout_returns_milliseconds_remaining_and_does_not_send_timeout_message() {
    with_timer(|milliseconds, message, timer_reference, process| {
        let half_milliseconds = milliseconds / 2;

        thread::sleep(Duration::from_millis(half_milliseconds + 1));
        timer::timeout();

        let timeout_message = timeout_message(timer_reference, message, process);

        assert!(!has_message(process, timeout_message));

        assert_eq!(
            erlang::read_timer_2(timer_reference, options(process), process),
            Ok(Term::str_to_atom("ok", DoNotCare).unwrap())
        );

        let first_received_message = receive_message(process).unwrap();

        assert_eq!(first_received_message.tag(), Boxed);

        let unboxed_first_received_message: &Term = first_received_message.unbox_reference();

        assert_eq!(unboxed_first_received_message.tag(), Arity);

        let first_received_tuple: &Tuple = first_received_message.unbox_reference();

        assert_eq!(
            first_received_tuple[0],
            Term::str_to_atom("read_timer", DoNotCare).unwrap()
        );
        assert_eq!(first_received_tuple[1], timer_reference);

        let first_milliseconds_remaining = first_received_tuple[2];

        assert!(first_milliseconds_remaining.is_integer());
        assert!(0.into_process(process) < first_milliseconds_remaining);
        assert!(first_milliseconds_remaining <= (milliseconds / 2).into_process(process));

        // again before timeout
        assert_eq!(
            erlang::read_timer_2(timer_reference, options(process), process),
            Ok(Term::str_to_atom("ok", DoNotCare).unwrap())
        );

        let second_received_message = receive_message(process).unwrap();

        assert_eq!(second_received_message.tag(), Boxed);

        let unboxed_second_received_message: &Term = second_received_message.unbox_reference();

        assert_eq!(unboxed_second_received_message.tag(), Arity);

        let second_received_tuple: &Tuple = second_received_message.unbox_reference();

        assert_eq!(
            second_received_tuple[0],
            Term::str_to_atom("read_timer", DoNotCare).unwrap()
        );
        assert_eq!(second_received_tuple[1], timer_reference);

        let second_milliseconds_remaining = second_received_tuple[2];

        assert!(second_milliseconds_remaining.is_integer());
        assert!(second_milliseconds_remaining <= first_milliseconds_remaining);

        thread::sleep(Duration::from_millis(half_milliseconds + 1));
        timer::timeout();

        assert_eq!(receive_message(process), Some(timeout_message));

        let false_read_timer_message = read_timer_message(timer_reference, false.into(), process);

        // again after timeout
        assert_eq!(
            erlang::read_timer_2(timer_reference, options(process), process),
            Ok(Term::str_to_atom("ok", DoNotCare).unwrap())
        );
        assert_eq!(receive_message(process), Some(false_read_timer_message));
    })
}

#[test]
fn with_timeout_returns_false_after_timeout_message_was_sent() {
    with_timer(|milliseconds, message, timer_reference, process| {
        thread::sleep(Duration::from_millis(milliseconds + 1));
        timer::timeout();

        let timeout_message = timeout_message(timer_reference, message, process);

        assert_eq!(receive_message(process), Some(timeout_message));

        let read_timer_message = read_timer_message(timer_reference, false.into(), process);

        assert_eq!(
            erlang::read_timer_2(timer_reference, options(process), process),
            Ok(Term::str_to_atom("ok", DoNotCare).unwrap())
        );
        assert_eq!(receive_message(process), Some(read_timer_message));

        // again
        assert_eq!(
            erlang::read_timer_2(timer_reference, options(process), process),
            Ok(Term::str_to_atom("ok", DoNotCare).unwrap())
        );
        assert_eq!(receive_message(process), Some(read_timer_message));
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
