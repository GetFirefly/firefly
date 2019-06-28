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
                let time = milliseconds.into_process(&arc_process);
                let destination = registered_name();

                let result = erlang::start_timer_3(time, destination, message, arc_process.clone());

                prop_assert!(
                    result.is_ok(),
                    "Timer reference not returned.  Got {:?}",
                    result
                );

                let timer_reference = result.unwrap();

                prop_assert_eq!(timer_reference.tag(), Boxed);

                let unboxed_timer_reference: &Term = timer_reference.unbox_reference();

                prop_assert_eq!(unboxed_timer_reference.tag(), LocalReference);

                let timeout_message = timeout_message(timer_reference, message, &arc_process);

                prop_assert!(!has_message(&arc_process, timeout_message));

                thread::sleep(Duration::from_millis(milliseconds + 1));

                timer::timeout();

                prop_assert!(!has_message(&arc_process, timeout_message));

                Ok(())
            },
        )
        .unwrap();
}
