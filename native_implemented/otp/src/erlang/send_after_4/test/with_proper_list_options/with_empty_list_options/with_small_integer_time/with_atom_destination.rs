use super::*;

mod registered;

#[test]
fn unregistered_sends_nothing_when_timer_expires() {
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
                let time = arc_process.integer(milliseconds).unwrap();
                let destination = registered_name();

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

                prop_assert!(!has_message(&arc_process, message));

                Ok(())
            },
        )
        .unwrap();
}
