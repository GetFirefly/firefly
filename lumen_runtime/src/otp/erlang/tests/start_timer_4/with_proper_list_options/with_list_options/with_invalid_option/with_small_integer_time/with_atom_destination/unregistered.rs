use super::*;

use proptest::strategy::Strategy;

#[test]
fn errors_badarg() {
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
                let time = arc_process.integer(milliseconds).unwrap();
                let destination = registered_name();
                let options = options(&arc_process);

                prop_assert_eq!(
                    erlang::start_timer_4(time, destination, message, options, arc_process.clone()),
                    Err(badarg!().into())
                );

                Ok(())
            },
        )
        .unwrap();
}
