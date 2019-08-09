use super::*;

#[test]
fn errors_with_reason() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term(arc_process.clone()), |reason| {
                prop_assert_eq!(erlang::error_1(reason), Err(error!(reason).into()));

                Ok(())
            })
            .unwrap();
    });
}
