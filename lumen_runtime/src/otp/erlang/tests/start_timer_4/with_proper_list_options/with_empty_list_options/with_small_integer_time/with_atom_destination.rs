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
                let destination = registered_name();

                let time = milliseconds.into_process(&arc_process);

                let result =
                    erlang::start_timer_4(time, destination, message, OPTIONS, arc_process.clone());

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

                thread::sleep(Duration::from_millis(milliseconds + 1));

                timer::timeout();

                prop_assert!(!has_message(&arc_process, timeout_message));

                Ok(())
            },
        )
        .unwrap();
}
