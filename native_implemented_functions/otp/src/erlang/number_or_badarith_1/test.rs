use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use crate::erlang::number_or_badarith_1::result;
use crate::test::strategy;
use crate::test::with_process_arc;

#[test]
fn without_number_errors_badarith() {
    run!(
        |arc_process| strategy::term::is_not_number(arc_process.clone()),
        |number| {
            prop_assert_badarith!(
                result(number),
                format!("number ({}) is not an integer or a float", number)
            );

            Ok(())
        },
    );
}

#[test]
fn with_number_returns_number() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_number(arc_process.clone()), |number| {
                prop_assert_eq!(result(number), Ok(number));

                Ok(())
            })
            .unwrap();
    });
}
