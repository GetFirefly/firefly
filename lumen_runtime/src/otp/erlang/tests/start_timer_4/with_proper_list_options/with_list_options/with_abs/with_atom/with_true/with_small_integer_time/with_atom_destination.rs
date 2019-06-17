use super::*;

mod registered;

#[test]
fn unregistered_sends_nothing_when_timer_expires() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    milliseconds(),
                    strategy::term::heap_fragment_safe(arc_process.clone()),
                ),
                |(milliseconds, message)| {
                    let time = milliseconds.into_process(&arc_process);
                    let destination = registered_name();
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

                    prop_assert!(!has_message(&arc_process, timeout_message));

                    // No sleeping is necessary because timeout is in the past and so the timer will
                    // timeout at once

                    timer::timeout();

                    prop_assert!(!has_message(&arc_process, timeout_message));

                    Ok(())
                },
            )
            .unwrap();
    });
}
