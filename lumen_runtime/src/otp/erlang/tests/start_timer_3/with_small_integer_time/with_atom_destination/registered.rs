use super::*;

#[test]
fn with_different_process_sends_message_when_timer_expires() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(milliseconds(), strategy::term(arc_process.clone())),
                |(milliseconds, message)| {
                    let time = arc_process.integer(milliseconds).unwrap();

                    let destination_arc_process = process::test(&arc_process);
                    let destination = registered_name();

                    prop_assert_eq!(
                        erlang::register_2(
                            destination,
                            destination_arc_process.pid_term(),
                            arc_process.clone()
                        ),
                        Ok(true.into())
                    );

                    let result =
                        erlang::start_timer_3(time, destination, message, arc_process.clone());

                    prop_assert!(
                        result.is_ok(),
                        "Timer reference not returned.  Got {:?}",
                        result
                    );

                    let timer_reference = result.unwrap();

                    prop_assert!(timer_reference.is_local_reference());

                    let timeout_message = timeout_message(timer_reference, message, &arc_process);

                    prop_assert!(!has_message(&destination_arc_process, timeout_message));

                    thread::sleep(Duration::from_millis(milliseconds + 1));

                    timer::timeout();

                    prop_assert!(has_message(&destination_arc_process, timeout_message));

                    Ok(())
                },
            )
            .unwrap();
    });
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
                let time = arc_process.integer(milliseconds).unwrap();
                let destination = registered_name();

                prop_assert_eq!(
                    erlang::register_2(destination, arc_process.pid_term(), arc_process.clone()),
                    Ok(true.into())
                );

                let result = erlang::start_timer_3(time, destination, message, arc_process.clone());

                prop_assert!(
                    result.is_ok(),
                    "Timer reference not returned.  Got {:?}",
                    result
                );

                let timer_reference = result.unwrap();

                prop_assert!(timer_reference.is_local_reference());

                let timeout_message = timeout_message(timer_reference, message, &arc_process);

                prop_assert!(!has_message(&arc_process, timeout_message));

                thread::sleep(Duration::from_millis(milliseconds + 1));

                timer::timeout();

                prop_assert!(has_message(&arc_process, timeout_message));

                Ok(())
            },
        )
        .unwrap();
}
