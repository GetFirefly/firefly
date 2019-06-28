use super::*;

use proptest::strategy::Strategy;

#[test]
fn with_different_process_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    milliseconds(),
                    strategy::term::heap_fragment_safe(arc_process.clone()),
                    options(arc_process.clone()),
                ),
                |(milliseconds, message, options)| {
                    let time = milliseconds.into_process(&arc_process);

                    let destination_arc_process = process::local::test(&arc_process);
                    let destination = registered_name();

                    prop_assert_eq!(
                        erlang::register_2(
                            destination,
                            destination_arc_process.pid,
                            arc_process.clone()
                        ),
                        Ok(true.into())
                    );

                    prop_assert_eq!(
                        erlang::start_timer_4(
                            time,
                            destination,
                            message,
                            options,
                            arc_process.clone(),
                        ),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_process_errors_badarg() {
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

                prop_assert_eq!(
                    erlang::register_2(destination, arc_process.pid, arc_process.clone()),
                    Ok(true.into())
                );

                let time = milliseconds.into_process(&arc_process);

                prop_assert_eq!(
                    erlang::start_timer_4(time, destination, message, options, arc_process.clone()),
                    Err(badarg!())
                );

                Ok(())
            },
        )
        .unwrap();
}
