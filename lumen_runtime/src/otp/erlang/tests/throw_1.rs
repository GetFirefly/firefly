use super::*;

#[test]
fn throws_reason() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term(arc_process.clone()), |reason| {
                prop_assert_eq!(erlang::throw_1(reason), Err(throw!(reason)));

                Ok(())
            })
            .unwrap();
    });
}
