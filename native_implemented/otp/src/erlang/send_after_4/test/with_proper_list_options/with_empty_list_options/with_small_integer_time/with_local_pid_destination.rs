use super::*;

#[test]
fn with_different_process_sends_message_when_timer_expires() {
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
            let destination = destination_arc_process.pid_term();

            let start_time_in_milliseconds = freeze_timeout();

            let result = result(
                arc_process.clone(),
                time,
                destination,
                message,
                options(&arc_process),
            );

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
fn with_same_process_sends_message_when_timer_expires() {
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
            let destination = arc_process.pid_term();

            let start_time_in_milliseconds = freeze_timeout();

            let result = result(
                arc_process.clone(),
                time,
                destination,
                message,
                options(&arc_process),
            );

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

#[test]
fn without_process_sends_nothing_when_timer_expires() {
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
            let destination = Pid::next_term();

            let start_time_in_milliseconds = freeze_timeout();

            let result = result(
                arc_process.clone(),
                time,
                destination,
                message,
                options(&arc_process),
            );

            prop_assert!(
                result.is_ok(),
                "Timer reference not returned.  Got {:?}",
                result
            );

            let timer_reference = result.unwrap();

            prop_assert!(timer_reference.is_boxed_local_reference());

            freeze_at_timeout(start_time_in_milliseconds + milliseconds + 1);

            // does not send to original process either
            prop_assert!(!has_message(&arc_process, message));

            Ok(())
        },
    );
}
