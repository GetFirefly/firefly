use super::*;

#[test]
fn without_boolean_right_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_boolean(arc_process.clone()),
                |right_boolean| {
                    prop_assert_is_not_boolean!(native(false.into(), right_boolean), right_boolean);

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_false_right_returns_false() {
    assert_eq!(native(false.into(), false.into()), Ok(false.into()));
}

#[test]
fn with_true_right_returns_true() {
    assert_eq!(native(false.into(), true.into()), Ok(true.into()));
}
