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
                    abs_value(arc_process.clone()),
                ),
                |(milliseconds, message, abs_value)| {
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
                    abs_value(arc_process),
                )
            }),
            |(milliseconds, arc_process, message, abs_value)| {
                let destination = registered_name();

                prop_assert_eq!(
                    erlang::register_2::native(
                        arc_process.clone(),
                        destination,
                        arc_process.pid_term()
                    ),
                    Ok(true.into())
                );

                let time = arc_process.integer(milliseconds).unwrap();
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
