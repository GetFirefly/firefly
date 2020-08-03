use proptest::arbitrary::any;
use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestRunner};

use crate::erlang::list_to_float_1::result;
use crate::test::strategy;
use crate::test::with_process_arc;

#[test]
fn without_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_list(arc_process.clone()), |list| {
                prop_assert_badarg!(
                    result(&arc_process, list),
                    format!("list ({}) is not a a list", list)
                );

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_list_with_integer_errors_badarg() {
    run!(
        |arc_process| {
            (Just(arc_process.clone()), any::<isize>()).prop_map(|(arc_process, integer)| {
                let string = integer.to_string();

                (
                    arc_process.clone(),
                    string.clone(),
                    arc_process.charlist_from_str(&string).unwrap(),
                )
            })
        },
        |(arc_process, string, list)| {
            prop_assert_badarg!(
                result(&arc_process, list),
                format!("list ('{}') does not contain decimal point", string)
            );

            Ok(())
        },
    );
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
                    result(&arc_process, list),
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

        assert_badarg!(
            result(&arc_process, list),
            "Erlang does not support infinities"
        );
    });
}

#[test]
fn with_list_with_greater_than_max_f64_errors_badarg() {
    with_process_arc(|arc_process| {
        let list = arc_process.charlist_from_str("1797693134862315700000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.0").unwrap();

        assert_badarg!(
            result(&arc_process, list),
            "Erlang does not support infinities"
        );
    });
}
