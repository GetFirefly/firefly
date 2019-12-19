use super::*;

#[test]
fn with_different_process_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    milliseconds(),
                    strategy::term(arc_process.clone()),
                    abs_value(arc_process.clone()),
                ),
                |(milliseconds, message, abs_value)| {
                    let time = arc_process.integer(milliseconds).unwrap();

                    let destination_arc_process = process::test(&arc_process);
                    let destination = destination_arc_process.pid_term();
                    let options = options(abs_value, &arc_process);

                    prop_assert_is_not_boolean!(
                        native(arc_process.clone(), time, destination, message, options),
                        "abs value",
                        abs_value
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
                    abs_value(arc_process.clone()),
                )
            }),
            |(milliseconds, arc_process, message, abs_value)| {
                let time = arc_process.integer(milliseconds).unwrap();
                let destination = arc_process.pid_term();
                let options = options(abs_value, &arc_process);

                prop_assert_is_not_boolean!(
                    native(arc_process.clone(), time, destination, message, options),
                    "abs value",
                    abs_value
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
                    abs_value(arc_process.clone()),
                ),
                |(milliseconds, message, abs_value)| {
                    let time = arc_process.integer(milliseconds).unwrap();
                    let destination = Pid::next_term();
                    let options = options(abs_value, &arc_process);

                    prop_assert_is_not_boolean!(
                        native(arc_process.clone(), time, destination, message, options),
                        "abs value",
                        abs_value
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
