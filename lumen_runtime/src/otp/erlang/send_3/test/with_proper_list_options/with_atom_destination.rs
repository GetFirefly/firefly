use super::*;

mod registered;

#[test]
fn unregistered_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()),
                    valid_options(arc_process.clone()),
                ),
                |(message, options)| {
                    let destination = registered_name();

                    prop_assert_badarg!(
                        native(&arc_process, destination, message, options),
                        format!("name ({}) not registered", destination)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
