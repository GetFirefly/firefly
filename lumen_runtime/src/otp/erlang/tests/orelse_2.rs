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
                    prop_assert_eq!(erlang::orelse_2(left, right), Err(badarg!()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_false_left_returns_right() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term(arc_process.clone()), |right| {
                prop_assert_eq!(erlang::orelse_2(false.into(), right), Ok(right));

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_true_left_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term(arc_process.clone()), |right| {
                prop_assert_eq!(erlang::orelse_2(true.into(), right), Ok(true.into()));

                Ok(())
            })
            .unwrap();
    });
}
