use super::*;

#[test]
fn without_boolean_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_boolean(arc_process.clone()),
                |boolean| {
                    prop_assert_eq!(erlang::not_1(boolean), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_false_returns_true() {
    assert_eq!(erlang::not_1(false.into()), Ok(true.into()));
}

#[test]
fn with_true_returns_false() {
    assert_eq!(erlang::not_1(true.into()), Ok(false.into()));
}
