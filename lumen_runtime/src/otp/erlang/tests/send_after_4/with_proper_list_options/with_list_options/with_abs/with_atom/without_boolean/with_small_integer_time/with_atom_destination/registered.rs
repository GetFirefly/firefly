use super::*;

use proptest::strategy::Strategy;

#[test]
fn with_different_process_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    milliseconds(),
                    strategy::term(arc_process.clone()),
                    options(arc_process.clone()),
                ),
                |(milliseconds, message, options)| {
                    let time = arc_process.integer(milliseconds).unwrap();

                    let destination_arc_process = process::test(&arc_process);
                    let destination = registered_name();

                    prop_assert_eq!(
                        erlang::register_2::native(
                            arc_process.clone(),
                            destination,
                            destination_arc_process.pid_term(),
                        ),
                        Ok(true.into())
                    );

                    prop_assert_eq!(
                        erlang::send_after_4(
                            time,
                            destination,
                            message,
                            options,
                            arc_process.clone(),
                        ),
                        Err(badarg!().into())
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
                    strategy::term(arc_process.clone()),
                    options(arc_process),
                )
            }),
            |(milliseconds, arc_process, message, options)| {
                let destination = registered_name();

                prop_assert_eq!(
                    erlang::register_2::native(
                        arc_process.clone(),
                        destination,
                        arc_process.pid_term(),
                    ),
                    Ok(true.into())
                );

                let time = arc_process.integer(milliseconds).unwrap();

                prop_assert_eq!(
                    erlang::send_after_4(time, destination, message, options, arc_process.clone()),
                    Err(badarg!().into())
                );

                Ok(())
            },
        )
        .unwrap();
}
