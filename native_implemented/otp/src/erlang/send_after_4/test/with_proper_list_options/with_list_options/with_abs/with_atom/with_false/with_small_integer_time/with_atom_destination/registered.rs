use super::*;

#[test]
fn with_different_process_with_message_sends_message_when_timer_expires() {
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

            let destination_arc_process = test::process::child(&arc_process);
            let destination = registered_name();

            prop_assert_eq!(
                erlang::register_2::result(
                    arc_process.clone(),
                    destination,
                    destination_arc_process.pid_term(),
                ),
                Ok(true.into())
            );

            let options = options(&arc_process);

            let start_time_in_milliseconds = freeze_timeout();

            let result = result(arc_process.clone(), time, destination, message, options);

            prop_assert!(
                result.is_ok(),
                "Timer reference not returned.  Got {:?}",
                result
            );

            let timer_reference = result.unwrap();

            prop_assert!(timer_reference.is_boxed_local_reference());
            prop_assert!(!has_message(&destination_arc_process, message));

            freeze_at_timeout(start_time_in_milliseconds + milliseconds + 1);

            prop_assert!(has_message(&destination_arc_process, message));

            Ok(())
        },
    );
}

#[test]
fn with_same_process_with_message_sends_message_when_timer_expires() {
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

            prop_assert_eq!(
                erlang::register_2::result(
                    arc_process.clone(),
                    destination,
                    arc_process.pid_term(),
                ),
                Ok(true.into())
            );

            let options = options(&arc_process);

            let start_time_in_milliseconds = freeze_timeout();

            let result = result(arc_process.clone(), time, destination, message, options);

            prop_assert!(
                result.is_ok(),
                "Timer reference not returned.  Got {:?}",
                result
            );

            let timer_reference = result.unwrap();

            prop_assert!(timer_reference.is_boxed_local_reference());
            prop_assert!(!has_message(&arc_process, message));

            freeze_at_timeout(start_time_in_milliseconds + milliseconds + 1);

            prop_assert!(has_message(&arc_process, message));

            Ok(())
        },
    );
}
