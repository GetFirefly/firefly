use super::*;

#[test]
fn errors_with_reason_and_arguments() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                ),
                |(reason, arguments)| {
                    prop_assert_eq!(
                        erlang::error_2(reason, arguments),
                        Err(error!(reason, Some(arguments)).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
