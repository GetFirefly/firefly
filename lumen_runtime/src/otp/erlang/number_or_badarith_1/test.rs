use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use crate::otp::erlang::number_or_badarith_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_number_errors_badarith() {
    run!(
        |arc_process| strategy::term::is_not_number(arc_process.clone()),
        |number| {
            prop_assert_badarith!(
                native(number),
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
                prop_assert_eq!(native(number), Ok(number));

                Ok(())
            })
            .unwrap();
    });
}
