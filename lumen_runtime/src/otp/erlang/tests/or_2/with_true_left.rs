use super::*;

#[test]
fn without_boolean_right_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_boolean(arc_process.clone()),
                |right| {
                    prop_assert_eq!(erlang::or_2(true.into(), right), Err(badarg!()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_boolean_right_returns_true() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(&strategy::term::is_boolean(), |right| {
            prop_assert_eq!(erlang::or_2(true.into(), right), Ok(true.into()));

            Ok(())
        })
        .unwrap();
}
