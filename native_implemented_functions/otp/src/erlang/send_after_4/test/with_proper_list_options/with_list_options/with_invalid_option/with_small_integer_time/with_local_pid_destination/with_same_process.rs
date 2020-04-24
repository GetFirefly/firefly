use super::*;

#[test]
fn sends_message_when_timer_expires() {
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
                let destination = arc_process.pid_term();
                let time = arc_process.integer(milliseconds).unwrap();
                let options = options(&arc_process);

                prop_assert_badarg!(
                    result(arc_process.clone(), time, destination, message, options),
                    "supported option is {:abs, bool}"
                );

                Ok(())
            },
        )
        .unwrap();
}
