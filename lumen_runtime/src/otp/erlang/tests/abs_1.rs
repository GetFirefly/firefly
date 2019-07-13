use super::*;

#[test]
fn without_number_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_number(arc_process.clone()),
                |number| {
                    prop_assert_eq!(erlang::abs_1(number, &arc_process), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_number_returns_non_negative() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_number(arc_process.clone()), |number| {
                let result = erlang::abs_1(number, &arc_process);

                prop_assert!(result.is_ok());

                let abs = result.unwrap();

                prop_assert!(arc_process.integer(0) <= abs);

                Ok(())
            })
            .unwrap();
    });
}
