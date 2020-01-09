use super::*;

use std::sync::Barrier;
use std::thread;
use std::time::Duration;

use crate::test::*;

#[test]
#[ignore]
fn without_timeout_returns_milliseconds_remaining() {
    with_timer_in_different_thread(|milliseconds, barrier, timer_reference, process| {
        timeout_after_half(milliseconds, barrier);

        let message = Atom::str_to_term("different");
        let timeout_message = timeout_message(timer_reference, message, process);

        assert!(!has_message(process, timeout_message));

        let first_milliseconds_remaining = erlang::read_timer_1::native(process, timer_reference)
            .expect("Timer could not be read");

        assert!(first_milliseconds_remaining.is_integer());
        assert!(process.integer(0).unwrap() < first_milliseconds_remaining);
        assert!(first_milliseconds_remaining <= process.integer(milliseconds / 2).unwrap());

        // again before timeout
        let second_milliseconds_remaining = erlang::read_timer_1::native(process, timer_reference)
            .expect("Timer could not be read");

        assert!(second_milliseconds_remaining.is_integer());
        assert!(second_milliseconds_remaining <= first_milliseconds_remaining);

        timeout_after_half(milliseconds, barrier);

        assert!(has_message(process, timeout_message));

        // again after timeout
        assert_eq!(
            erlang::read_timer_1::native(process, timer_reference,),
            Ok(false.into())
        );
    });
}

#[test]
fn with_timeout_returns_false() {
    with_timer_in_different_thread(|milliseconds, barrier, timer_reference, process| {
        timeout_after_half(milliseconds, barrier);
        timeout_after_half(milliseconds, barrier);

        let message = Atom::str_to_term("different");
        let timeout_message = timeout_message(timer_reference, message, process);

        assert!(
            has_message(process, timeout_message),
            "Mailbox does not contain {:?} and instead contains {:?}",
            timeout_message,
            process.mailbox.lock().borrow()
        );

        assert_eq!(native(process, timer_reference, options(process)), Ok(false.into()));

        // again
        assert_eq!(native(process, timer_reference, options(process)), Ok(false.into()));
    });
}

fn timeout_after_half(milliseconds: Milliseconds, barrier: &Barrier) {
    thread::sleep(Duration::from_millis(milliseconds / 2 + 1));
    timer::timeout();
    barrier.wait();
}
