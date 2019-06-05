use super::*;

#[test]
fn without_integer_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_integer(arc_process.clone()),
                |term| {
                    prop_assert_eq!(erlang::is_integer_1(term), false.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_integer_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_integer(arc_process.clone()), |term| {
                prop_assert_eq!(erlang::is_integer_1(term), true.into());

                Ok(())
            })
            .unwrap();
    });
}
