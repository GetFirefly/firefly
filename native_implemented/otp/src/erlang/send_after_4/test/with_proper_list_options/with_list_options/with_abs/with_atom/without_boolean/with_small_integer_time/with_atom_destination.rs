use super::*;

mod registered;

#[test]
fn unregistered_errors_badarg() {
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
                let time = arc_process.integer(milliseconds).unwrap();
                let options = options(abs_value, &arc_process);

                prop_assert_is_not_boolean!(
                    result(arc_process.clone(), time, destination, message, options),
                    "abs value",
                    abs_value
                );

                Ok(())
            },
        )
        .unwrap();
}
