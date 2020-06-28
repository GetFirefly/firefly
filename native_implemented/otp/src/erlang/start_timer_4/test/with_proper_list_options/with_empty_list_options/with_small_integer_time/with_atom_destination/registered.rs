use super::*;
use crate::test::freeze_timeout;

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

            let time = arc_process.integer(milliseconds).unwrap();

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

            let timeout_message = arc_process
                .tuple_from_slice(&[Atom::str_to_term("timeout"), timer_reference, message])
                .unwrap();

            prop_assert!(!has_message(&destination_arc_process, timeout_message));

            freeze_at_timeout(start_time_in_milliseconds + milliseconds + 1);

            prop_assert!(has_message(&destination_arc_process, timeout_message));

            Ok(())
        },
    );
}

#[test]
fn with_same_process_sends_message_when_timer_expires() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &(milliseconds(), strategy::process()).prop_flat_map(|(milliseconds, arc_process)| {
                (
                    Just(milliseconds),
                    Just(arc_process.clone()),
                    strategy::term(arc_process),
                )
            }),
            |(milliseconds, arc_process, message)| {
                let destination = registered_name();

                prop_assert_eq!(
                    erlang::register_2::result(
                        arc_process.clone(),
                        destination,
                        arc_process.pid_term(),
                    ),
                    Ok(true.into())
                );

                let time = arc_process.integer(milliseconds).unwrap();

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

                let timeout_message = arc_process
                    .tuple_from_slice(&[Atom::str_to_term("timeout"), timer_reference, message])
                    .unwrap();

                prop_assert!(!has_message(&arc_process, timeout_message));

                freeze_at_timeout(start_time_in_milliseconds + milliseconds + 1);

                prop_assert!(has_message(&arc_process, timeout_message));

                Ok(())
            },
        )
        .unwrap();
}
