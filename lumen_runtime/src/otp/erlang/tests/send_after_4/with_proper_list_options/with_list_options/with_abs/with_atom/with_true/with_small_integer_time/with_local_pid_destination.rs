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
                    strategy::term::heap_fragment_safe(arc_process),
                )
            }),
            |(milliseconds, arc_process, message)| {
                let time = milliseconds.into_process(&arc_process);
                let destination = registered_name();
                let options = options(&arc_process);

                let result =
                    erlang::send_after_4(time, destination, message, options, arc_process.clone());

                assert!(
                    result.is_ok(),
                    "Timer reference not returned.  Got {:?}",
                    result
                );

                let timer_reference = result.unwrap();

                assert_eq!(timer_reference.tag(), Boxed);

                let unboxed_timer_reference: &Term = timer_reference.unbox_reference();

                assert_eq!(unboxed_timer_reference.tag(), LocalReference);

                assert!(!has_message(&arc_process, message));

                // No sleeping is necessary because timeout is in the past and so the timer will
                // timeout at once

                timer::timeout();

                assert!(!has_message(&arc_process, message));

                Ok(())
            },
        )
        .unwrap();
}
