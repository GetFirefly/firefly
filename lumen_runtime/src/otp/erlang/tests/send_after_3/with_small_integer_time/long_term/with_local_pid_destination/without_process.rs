use super::*;

use std::thread;
use std::time::Duration;

#[test]
fn with_small_integer_message_does_not_panic_when_timer_expires() {
    with_message_does_not_panic_when_timer_expires(|process| 0.into_process(process));
}

#[test]
fn with_float_message_does_not_panic_when_timer_expires() {
    with_message_does_not_panic_when_timer_expires(|process| 1.0.into_process(process));
}

#[test]
fn with_big_integer_message_does_not_panic_when_timer_expires() {
    with_message_does_not_panic_when_timer_expires(|process| {
        (integer::small::MAX + 1).into_process(process)
    });
}

#[test]
fn with_local_reference_message_does_not_panic_when_timer_expires() {
    with_message_does_not_panic_when_timer_expires(|process| Term::next_local_reference(process));
}

#[test]
fn with_external_pid_message_does_not_panic_when_timer_expires() {
    with_message_does_not_panic_when_timer_expires(|process| {
        Term::external_pid(1, 0, 0, process).unwrap()
    });
}

#[test]
fn with_tuple_message_does_not_panic_when_timer_expires() {
    with_message_does_not_panic_when_timer_expires(|process| Term::slice_to_tuple(&[], process));
}

#[test]
fn with_map_message_does_not_panic_when_timer_expires() {
    with_message_does_not_panic_when_timer_expires(|process| Term::slice_to_map(&[], process));
}

#[test]
fn with_empty_list_message_does_not_panic_when_timer_expires() {
    with_message_does_not_panic_when_timer_expires(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_message_does_not_panic_when_timer_expires() {
    with_message_does_not_panic_when_timer_expires(|process| list_term(process));
}

#[test]
fn with_heap_binary_message_does_not_panic_when_timer_expires() {
    with_message_does_not_panic_when_timer_expires(|process| Term::slice_to_binary(&[1], process));
}

#[test]
fn with_subbinary_message_does_not_panic_when_timer_expires() {
    with_message_does_not_panic_when_timer_expires(|process| bitstring!(1 :: 1, process));
}

fn with_message_does_not_panic_when_timer_expires<M>(message: M)
where
    M: FnOnce(&Process) -> Term,
{
    with_process_arc(|process_arc| {
        let destination = process::identifier::local::next();

        let milliseconds = milliseconds();
        let time = milliseconds.into_process(&process_arc);
        let message = message(&process_arc);

        let result = erlang::send_after_3(time, destination, message, process_arc.clone());

        assert!(
            result.is_ok(),
            "Timer reference not returned.  Got {:?}",
            result
        );

        let timer_reference = result.unwrap();

        assert_eq!(timer_reference.tag(), Boxed);

        let unboxed_timer_reference: &Term = timer_reference.unbox_reference();

        assert_eq!(unboxed_timer_reference.tag(), LocalReference);

        thread::sleep(Duration::from_millis(milliseconds + 1));
        timer::timeout();

        // does not send to original process either
        assert!(!has_message(&process_arc, message));
    })
}
