use super::*;

#[test]
fn with_different_process_with_message_sends_message_when_timer_expires() {
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

                    prop_assert_eq!(
                        erlang::register_2(
                            destination,
                            destination_arc_process.pid,
                            arc_process.clone()
                        ),
                        Ok(true.into())
                    );

                    let options = options(&arc_process);

                    let result = erlang::start_timer_4(
                        time,
                        destination,
                        message,
                        options,
                        arc_process.clone(),
                    );

                    prop_assert!(
                        result.is_ok(),
                        "Timer reference not returned.  Got {:?}",
                        result
                    );

                    let timer_reference = result.unwrap();

                    prop_assert_eq!(timer_reference.tag(), Boxed);

                    let unboxed_timer_reference: &Term = timer_reference.unbox_reference();

                    prop_assert_eq!(unboxed_timer_reference.tag(), LocalReference);
                    let timeout_message = Term::slice_to_tuple(
                        &[
                            Term::str_to_atom("timeout", DoNotCare).unwrap(),
                            timer_reference,
                            message,
                        ],
                        &arc_process,
                    );

                    prop_assert!(!has_message(&destination_arc_process, timeout_message));

                    // No sleeping is necessary because timeout is in the past and so the timer will
                    // timeout at once

                    timer::timeout();

                    prop_assert!(has_message(&destination_arc_process, timeout_message));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_process_with_message_sends_message_when_timer_expires() {
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

                prop_assert_eq!(
                    erlang::register_2(destination, arc_process.pid, arc_process.clone()),
                    Ok(true.into())
                );

                let options = options(&arc_process);

                let result =
                    erlang::start_timer_4(time, destination, message, options, arc_process.clone());

                prop_assert!(
                    result.is_ok(),
                    "Timer reference not returned.  Got {:?}",
                    result
                );

                let timer_reference = result.unwrap();

                prop_assert_eq!(timer_reference.tag(), Boxed);

                let unboxed_timer_reference: &Term = timer_reference.unbox_reference();

                prop_assert_eq!(unboxed_timer_reference.tag(), LocalReference);

                let timeout_message = Term::slice_to_tuple(
                    &[
                        Term::str_to_atom("timeout", DoNotCare).unwrap(),
                        timer_reference,
                        message,
                    ],
                    &arc_process,
                );

                prop_assert!(!has_message(&arc_process, timeout_message));

                // No sleeping is necessary because timeout is in the past and so the timer will
                // timeout at once

                timer::timeout();

                prop_assert!(has_message(&arc_process, timeout_message));

                Ok(())
            },
        )
        .unwrap();
}
