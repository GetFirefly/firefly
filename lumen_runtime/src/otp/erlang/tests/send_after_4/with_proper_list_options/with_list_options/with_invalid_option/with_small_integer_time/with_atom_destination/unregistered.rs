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
                    strategy::term::heap_fragment_safe(arc_process),
                )
            }),
            |(milliseconds, arc_process, message)| {
                let time = milliseconds.into_process(&arc_process);
                let destination = registered_name();
                let options = options(&arc_process);

                prop_assert_eq!(
                    erlang::send_after_4(time, destination, message, options, arc_process.clone()),
                    Err(badarg!())
                );

                Ok(())
            },
        )
        .unwrap();
}
