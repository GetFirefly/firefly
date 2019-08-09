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
                    options(arc_process),
                )
            }),
            |(milliseconds, arc_process, message, options)| {
                let destination = registered_name();

                let time = arc_process.integer(milliseconds).unwrap();

                prop_assert_eq!(
                    erlang::send_after_4(time, destination, message, options, arc_process.clone(),),
                    Err(badarg!().into())
                );

                Ok(())
            },
        )
        .unwrap();
}
