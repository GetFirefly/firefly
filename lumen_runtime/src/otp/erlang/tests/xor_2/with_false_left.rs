use super::*;

#[test]
fn without_boolean_right_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_boolean(arc_process.clone()),
                |right| {
                    prop_assert_eq!(erlang::xor_2(false.into(), right), Err(badarg!()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_false_right_returns_false() {
    assert_eq!(erlang::xor_2(false.into(), false.into()), Ok(false.into()));
}

#[test]
fn with_true_right_returns_true() {
    assert_eq!(erlang::xor_2(false.into(), true.into()), Ok(true.into()));
}
