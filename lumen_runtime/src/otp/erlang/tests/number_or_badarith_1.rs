use super::*;

#[test]
fn without_number_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_number(arc_process.clone()),
                |number| {
                    prop_assert_eq!(erlang::number_or_badarith_1(number), Err(badarith!()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_number_returns_number() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_number(arc_process.clone()), |number| {
                prop_assert_eq!(erlang::number_or_badarith_1(number), Ok(number));

                Ok(())
            })
            .unwrap();
    });
}
