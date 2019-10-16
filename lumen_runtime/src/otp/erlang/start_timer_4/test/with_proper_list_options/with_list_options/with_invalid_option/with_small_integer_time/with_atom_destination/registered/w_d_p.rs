use super::*;

#[test]
fn errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(milliseconds(), strategy::term(arc_process.clone())),
                |(milliseconds, message)| {
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

                    let options = options(&arc_process);

                    prop_assert_eq!(
                        native(arc_process.clone(), time, destination, message, options),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
