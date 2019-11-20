use super::*;

#[test]
fn without_boolean_right_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_boolean(arc_process.clone()),
                |right| {
                    prop_assert_eq!(
                        native(&arc_process, true.into(), right),
                        Err(badarg!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_boolean_right_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_boolean(), |right| {
                prop_assert_eq!(native(&arc_process, true.into(), right), Ok(true.into()));

                Ok(())
            })
            .unwrap();
    });
}
