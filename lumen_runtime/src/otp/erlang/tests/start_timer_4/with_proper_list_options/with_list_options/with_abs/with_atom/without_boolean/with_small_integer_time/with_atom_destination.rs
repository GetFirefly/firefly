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
                    strategy::term::heap_fragment_safe(arc_process.clone()),
                    options(arc_process),
                )
            }),
            |(milliseconds, arc_process, message, options)| {
                let destination = registered_name();

                let time = milliseconds.into_process(&arc_process);

                prop_assert_eq!(
                    erlang::start_timer_4(time, destination, message, options, arc_process.clone(),),
                    Err(badarg!())
                );

                Ok(())
            },
        )
        .unwrap();
}
