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
                let destination = registered_name();

                let time = arc_process.integer(milliseconds).unwrap();

                let result = native(arc_process.clone(), time, destination, message, OPTIONS);

                prop_assert!(
                    result.is_ok(),
                    "Timer reference not returned.  Got {:?}",
                    result
                );

                let timer_reference = result.unwrap();

                prop_assert!(timer_reference.is_local_reference());

                let timeout_message = arc_process
                    .tuple_from_slice(&[atom_unchecked("timeout"), timer_reference, message])
                    .unwrap();

                prop_assert!(!has_message(&arc_process, timeout_message));

                thread::sleep(Duration::from_millis(milliseconds + 1));

                timer::timeout();

                prop_assert!(!has_message(&arc_process, timeout_message));

                Ok(())
            },
        )
        .unwrap();
}
