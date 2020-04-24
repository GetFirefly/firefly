use super::*;

mod registered;

#[test]
fn unregistered_sends_nothing_when_timer_expires() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                milliseconds(),
                strategy::term(arc_process.clone()),
            )
        },
        |(arc_process, milliseconds, message)| {
            let time = arc_process.integer(milliseconds).unwrap();
            let destination = registered_name();
            let options = options(&arc_process);

            let result = result(arc_process.clone(), time, destination, message, options);

            prop_assert!(
                result.is_ok(),
                "Timer reference not returned.  Got {:?}",
                result
            );

            let timer_reference = result.unwrap();

            prop_assert!(timer_reference.is_boxed_local_reference());

            let timeout_message = arc_process
                .tuple_from_slice(&[Atom::str_to_term("timeout"), timer_reference, message])
                .unwrap();

            prop_assert!(!has_message(&arc_process, timeout_message));

            // No sleeping is necessary because timeout is in the past and so the timer will
            // timeout at once

            crate::runtime::timer::timeout();

            prop_assert!(!has_message(&arc_process, timeout_message));

            Ok(())
        },
    );
}
