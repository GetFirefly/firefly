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
                    let time = milliseconds.into_process(&arc_process);

                    let destination_arc_process = process::local::test(&arc_process);
                    let destination = registered_name();

                    assert_eq!(
                        erlang::register_2(
                            destination,
                            destination_arc_process.pid,
                            arc_process.clone()
                        ),
                        Ok(true.into())
                    );

                    let result =
                        erlang::start_timer_3(time, destination, message, arc_process.clone());

                    assert!(
                        result.is_ok(),
                        "Timer reference not returned.  Got {:?}",
                        result
                    );

                    let timer_reference = result.unwrap();

                    assert_eq!(timer_reference.tag(), Boxed);

                    let unboxed_timer_reference: &Term = timer_reference.unbox_reference();

                    assert_eq!(unboxed_timer_reference.tag(), LocalReference);

                    let timeout_message = timeout_message(timer_reference, message, &arc_process);

                    assert!(!has_message(&destination_arc_process, timeout_message));

                    thread::sleep(Duration::from_millis(milliseconds + 1));

                    timer::timeout();

                    assert!(has_message(&destination_arc_process, timeout_message));

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
                let time = milliseconds.into_process(&arc_process);
                let destination = registered_name();

                assert_eq!(
                    erlang::register_2(destination, arc_process.pid, arc_process.clone()),
                    Ok(true.into())
                );

                let result = erlang::start_timer_3(time, destination, message, arc_process.clone());

                assert!(
                    result.is_ok(),
                    "Timer reference not returned.  Got {:?}",
                    result
                );

                let timer_reference = result.unwrap();

                assert_eq!(timer_reference.tag(), Boxed);

                let unboxed_timer_reference: &Term = timer_reference.unbox_reference();

                assert_eq!(unboxed_timer_reference.tag(), LocalReference);

                let timeout_message = timeout_message(timer_reference, message, &arc_process);

                assert!(!has_message(&arc_process, timeout_message));

                thread::sleep(Duration::from_millis(milliseconds + 1));

                timer::timeout();

                assert!(has_message(&arc_process, timeout_message));

                Ok(())
            },
        )
        .unwrap();
}
