use super::*;

#[test]
fn without_boolean_right_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_boolean(arc_process.clone()),
                |right| {
                    prop_assert_badarg!(
                        native(true.into(), right),
                        format!("right ({}) must be a bool", right)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_false_right_returns_false() {
    assert_eq!(native(true.into(), false.into()), Ok(false.into()));
}

#[test]
fn with_true_right_returns_true() {
    assert_eq!(native(true.into(), true.into()), Ok(true.into()));
}
