use super::*;

// `returns_binary` in integration tests
// `is_the_same_as_float_to_binary_2_with_scientific_20` in integration tests

#[test]
fn is_dual_of_binary_to_float_1() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::float(arc_process.clone()), |float| {
                let result_binary = result(&arc_process, float, options(&arc_process));

                prop_assert!(result_binary.is_ok());

                let binary = result_binary.unwrap();

                prop_assert_eq!(binary_to_float_1::result(&arc_process, binary), Ok(float));

                Ok(())
            })
            .unwrap();
    });
}

fn options(_: &Process) -> Term {
    Term::NIL
}
