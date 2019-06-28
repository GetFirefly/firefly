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
                    let destination = destination_arc_process.pid;

                    let result =
                        erlang::send_after_3(time, destination, message, arc_process.clone());

                    prop_assert!(
                        result.is_ok(),
                        "Timer reference not returned.  Got {:?}",
                        result
                    );

                    let timer_reference = result.unwrap();

                    prop_assert_eq!(timer_reference.tag(), Boxed);

                    let unboxed_timer_reference: &Term = timer_reference.unbox_reference();

                    prop_assert_eq!(unboxed_timer_reference.tag(), LocalReference);

                    prop_assert!(!has_message(&destination_arc_process, message));

                    thread::sleep(Duration::from_millis(milliseconds + 1));

                    timer::timeout();

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
                let time = milliseconds.into_process(&arc_process);

                let destination = arc_process.pid;

                let result = erlang::send_after_3(time, destination, message, arc_process.clone());

                prop_assert!(
                    result.is_ok(),
                    "Timer reference not returned.  Got {:?}",
                    result
                );

                let timer_reference = result.unwrap();

                prop_assert_eq!(timer_reference.tag(), Boxed);

                let unboxed_timer_reference: &Term = timer_reference.unbox_reference();

                prop_assert_eq!(unboxed_timer_reference.tag(), LocalReference);

                prop_assert!(!has_message(&arc_process, message));

                thread::sleep(Duration::from_millis(milliseconds + 1));
                timer::timeout();

                prop_assert!(has_message(&arc_process, message));

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn without_process_sends_nothing_when_timer_expires() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(milliseconds(), strategy::term(arc_process.clone())),
                |(milliseconds, message)| {
                    let destination = process::identifier::local::next();

                    let time = milliseconds.into_process(&arc_process);

                    let result =
                        erlang::send_after_3(time, destination, message, arc_process.clone());

                    prop_assert!(
                        result.is_ok(),
                        "Timer reference not returned.  Got {:?}",
                        result
                    );

                    let timer_reference = result.unwrap();

                    prop_assert_eq!(timer_reference.tag(), Boxed);

                    let unboxed_timer_reference: &Term = timer_reference.unbox_reference();

                    prop_assert_eq!(unboxed_timer_reference.tag(), LocalReference);

                    thread::sleep(Duration::from_millis(milliseconds + 1));
                    timer::timeout();

                    // does not send to original process either
                    prop_assert!(!has_message(&arc_process, message));

                    Ok(())
                },
            )
            .unwrap();
    });
}
