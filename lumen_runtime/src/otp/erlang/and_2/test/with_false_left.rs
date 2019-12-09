use super::*;

#[test]
fn without_boolean_right_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_boolean(arc_process.clone()),
                |right| {
                    prop_assert_badarg!(
                        native(false.into(), right),
                        format!("right ({}) must be a bool", right)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_boolean_right_returns_false() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(&strategy::term::is_boolean(), |right| {
            prop_assert_eq!(native(false.into(), right), Ok(false.into()));

            Ok(())
        })
        .unwrap();
}
