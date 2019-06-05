use super::*;

#[test]
fn without_pid_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_pid(arc_process.clone()), |term| {
                prop_assert_eq!(erlang::is_pid_1(term), false.into());

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_pid_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_pid(arc_process.clone()), |term| {
                prop_assert_eq!(erlang::is_pid_1(term), true.into());

                Ok(())
            })
            .unwrap();
    });
}
