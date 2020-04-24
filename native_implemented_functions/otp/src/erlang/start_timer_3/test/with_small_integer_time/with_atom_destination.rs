use super::*;

mod registered;

#[test]
fn unregistered_sends_nothing_when_timer_expires() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                milliseconds(),
                strategy::term(arc_process),
            )
        },
        |(arc_process, milliseconds, message)| {
            let time = arc_process.integer(milliseconds).unwrap();
            let destination = registered_name();

            let start_time_in_milliseconds = freeze_timeout();

            let result =
                erlang::start_timer_3::result(arc_process.clone(), time, destination, message);

            prop_assert!(
                result.is_ok(),
                "Timer reference not returned.  Got {:?}",
                result
            );

            let timer_reference = result.unwrap();

            prop_assert!(timer_reference.is_boxed_local_reference());

            let timeout_message = timeout_message(timer_reference, message, &arc_process);

            prop_assert!(!has_message(&arc_process, timeout_message));

            freeze_at_timeout(start_time_in_milliseconds + milliseconds + 1);

            prop_assert!(!has_message(&arc_process, timeout_message));

            Ok(())
        },
    );
}
