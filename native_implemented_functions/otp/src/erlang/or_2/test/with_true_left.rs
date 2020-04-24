use super::*;

#[test]
fn without_boolean_right_errors_badarg() {
    run!(
        |arc_process| strategy::term::is_not_boolean(arc_process.clone()),
        |right_boolean| {
            prop_assert_is_not_boolean!(result(true.into(), right_boolean), right_boolean);

            Ok(())
        },
    );
}

#[test]
fn with_boolean_right_returns_true() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(&strategy::term::is_boolean(), |right| {
            prop_assert_eq!(result(true.into(), right), Ok(true.into()));

            Ok(())
        })
        .unwrap();
}
