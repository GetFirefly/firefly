use super::*;

use proptest::arbitrary::any;

use crate::erlang::list_to_float_1;

#[test]
fn with_float_returns_list() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&any::<f64>(), |float_f64| {
                let float = arc_process.float(float_f64).unwrap();

                let result = result(&arc_process, float);

                prop_assert!(result.is_ok());

                let term = result.unwrap();

                prop_assert!(term.is_list());

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn is_the_same_as_float_to_list_2_with_scientific_20() {
    with_process_arc(|arc_process| {
        assert_eq!(
            result(&arc_process, arc_process.float(0.0).unwrap()),
            Ok(arc_process
                .charlist_from_str("0.00000000000000000000e+00")
                .unwrap())
        );
        assert_eq!(
            result(&arc_process, arc_process.float(0.1).unwrap()),
            Ok(arc_process
                .charlist_from_str("1.00000000000000005551e-01")
                .unwrap())
        );
    });
}

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
