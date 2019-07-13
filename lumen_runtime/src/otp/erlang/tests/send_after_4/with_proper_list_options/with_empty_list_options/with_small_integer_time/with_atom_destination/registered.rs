use super::*;

#[test]
fn with_different_process_sends_message_when_timer_expires() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    milliseconds(),
                    strategy::term::heap_fragment_safe(arc_process.clone()),
                ),
                |(milliseconds, message)| {
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

                    let time = arc_process.integer(milliseconds);

                    let result = erlang::send_after_4(
                        time,
                        destination,
                        message,
                        OPTIONS,
                        arc_process.clone(),
                    );

                    prop_assert!(
                        result.is_ok(),
                        "Timer reference not returned.  Got {:?}",
                        result
                    );

                    let timer_reference = result.unwrap();

                    prop_assert!(timer_reference.is_local_reference());
                    prop_assert!(!has_message(&destination_arc_process, message));

                    thread::sleep(Duration::from_millis(milliseconds + 1));

                    timer::timeout().unwrap();

                    prop_assert!(has_message(&destination_arc_process, message));

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
                    strategy::term::heap_fragment_safe(arc_process),
                )
            }),
            |(milliseconds, arc_process, message)| {
                let destination = registered_name();

                prop_assert_eq!(
                    erlang::register_2(destination, arc_process.pid_term(), arc_process.clone()),
                    Ok(true.into())
                );

                let time = arc_process.integer(milliseconds);

                let result =
                    erlang::send_after_4(time, destination, message, OPTIONS, arc_process.clone());

                prop_assert!(
                    result.is_ok(),
                    "Timer reference not returned.  Got {:?}",
                    result
                );

                let timer_reference = result.unwrap();

                prop_assert!(timer_reference.is_local_reference());
                prop_assert!(!has_message(&arc_process, message));

                thread::sleep(Duration::from_millis(milliseconds + 1));

                timer::timeout().unwrap();

                prop_assert!(has_message(&arc_process, message));

                Ok(())
            },
        )
        .unwrap();
}
