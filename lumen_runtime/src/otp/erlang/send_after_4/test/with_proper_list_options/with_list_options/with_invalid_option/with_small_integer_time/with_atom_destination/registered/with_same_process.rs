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

                prop_assert_eq!(
                    erlang::register_2::native(
                        arc_process.clone(),
                        destination,
                        arc_process.pid_term(),
                    ),
                    Ok(true.into())
                );

                let options = options(&arc_process);

                prop_assert_eq!(
                    native(arc_process.clone(), time, destination, message, options),
                    Err(badarg!().into())
                );

                Ok(())
            },
        )
        .unwrap();
}
