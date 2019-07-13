use super::*;

#[test]
fn without_boolean_left_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_boolean(arc_process.clone()),
                    strategy::term::is_boolean(),
                ),
                |(left, right)| {
                    prop_assert_eq!(erlang::andalso_2(left, right), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_false_left_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term(arc_process.clone()), |right| {
                prop_assert_eq!(erlang::andalso_2(false.into(), right), Ok(false.into()));

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_true_left_returns_right() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term(arc_process.clone()), |right| {
                prop_assert_eq!(erlang::andalso_2(true.into(), right), Ok(right));

                Ok(())
            })
            .unwrap();
    });
}
