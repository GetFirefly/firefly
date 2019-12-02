use super::*;

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
                    let destination = destination_arc_process.pid_term();

                    prop_assert_eq!(
                        native(arc_process.clone(), time, destination, message, options),
                        Err(badarg!(&arc_process).into())
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
                let time = arc_process.integer(milliseconds).unwrap();
                let destination = arc_process.pid_term();

                prop_assert_eq!(
                    native(arc_process.clone(), time, destination, message, options),
                    Err(badarg!(&arc_process).into())
                );

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn without_process_errors_badarg() {
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
                    let destination = Pid::next_term();

                    prop_assert_eq!(
                        native(arc_process.clone(), time, destination, message, options),
                        Err(badarg!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
