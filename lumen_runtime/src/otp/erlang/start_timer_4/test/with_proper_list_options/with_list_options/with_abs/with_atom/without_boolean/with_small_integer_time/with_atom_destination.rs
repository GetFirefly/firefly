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
                    strategy::term(arc_process.clone()),
                    abs_value(arc_process),
                )
            }),
            |(milliseconds, arc_process, message, abs_value)| {
                let destination = registered_name();
                let time = arc_process.integer(milliseconds).unwrap();
                let options = options(abs_value, &arc_process);

                prop_assert_badarg!(
                    native(arc_process.clone(), time, destination, message, options),
                    format!("abs value ({}) must be boolean", abs_value)
                );

                Ok(())
            },
        )
        .unwrap();
}
