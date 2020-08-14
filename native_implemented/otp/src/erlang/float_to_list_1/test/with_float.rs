use super::*;

use crate::erlang::list_to_float_1;

// `returns_list` in integration tests
// `is_the_same_as_float_to_list_2_with_scientific_20` in integration tests

#[test]
fn is_dual_of_list_to_float_1() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::float(arc_process.clone()), |float| {
                let result_list = result(&arc_process, float);

                prop_assert!(result_list.is_ok());

                let list = result_list.unwrap();

                prop_assert_eq!(list_to_float_1::result(&arc_process, list), Ok(float));

                Ok(())
            })
            .unwrap();
    });
}
