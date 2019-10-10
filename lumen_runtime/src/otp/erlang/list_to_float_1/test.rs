use proptest::arbitrary::any;
use proptest::prop_assert_eq;
use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarg;

use crate::otp::erlang::list_to_float_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_list(arc_process.clone()), |list| {
                prop_assert_eq!(native(&arc_process, list), Err(badarg!().into()));

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_list_with_integer_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &any::<isize>().prop_map(|integer| {
                    arc_process.charlist_from_str(&integer.to_string()).unwrap()
                }),
                |list| {
                    prop_assert_eq!(native(&arc_process, list), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_list_with_f64_returns_floats() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &(strategy::process(), any::<f64>()).prop_map(|(arc_process, f)| {
                (
                    arc_process.clone(),
                    f,
                    arc_process.charlist_from_str(&format!("{:?}", f)).unwrap(),
                )
            }),
            |(arc_process, f, list)| {
                prop_assert_eq!(
                    native(&arc_process, list),
                    Ok(arc_process.float(f).unwrap())
                );

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn with_list_with_less_than_min_f64_errors_badarg() {
    with_process_arc(|arc_process| {
        let list = arc_process.charlist_from_str("-1797693134862315700000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.0").unwrap();

        assert_eq!(native(&arc_process, list), Err(badarg!().into()));
    });
}

#[test]
fn with_list_with_greater_than_max_f64_errors_badarg() {
    with_process_arc(|arc_process| {
        let list = arc_process.charlist_from_str("1797693134862315700000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.0").unwrap();

        assert_eq!(native(&arc_process, list), Err(badarg!().into()));
    });
}
