use super::*;

use crate::scheduler::with_process;

#[test]
fn without_boolean_right_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_boolean(arc_process.clone()),
                |right| {
                    prop_assert_eq!(
                        native(&arc_process, false.into(), right),
                        Err(badarg!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_false_right_returns_false() {
    with_process(|process| {
        assert_eq!(
            native(process, false.into(), false.into()),
            Ok(false.into())
        );
    });
}

#[test]
fn with_true_right_returns_true() {
    with_process(|process| {
        assert_eq!(native(process, false.into(), true.into()), Ok(true.into()));
    });
}
