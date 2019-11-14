use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert_eq, prop_oneof};

use liblumen_alloc::badarith;

use crate::otp::erlang::negate_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_number_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_number(arc_process.clone()),
                |number| {
                    prop_assert_eq!(
                        native(&arc_process, number),
                        Err(badarith!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_integer_returns_integer_of_opposite_sign() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &prop_oneof![std::isize::MIN..=-1, 1..=std::isize::MAX]
                    .prop_map(|i| (arc_process.integer(i).unwrap(), i)),
                |(number, i)| {
                    let negated = arc_process.integer(-i).unwrap();

                    prop_assert_eq!(native(&arc_process, number), Ok(negated));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_float_returns_float_of_opposite_sign() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &prop_oneof![std::f64::MIN..=-1.0, 1.0..=std::f64::MAX]
                    .prop_map(|f| (arc_process.float(f).unwrap(), f)),
                |(number, f)| {
                    let negated = arc_process.float(-f).unwrap();

                    prop_assert_eq!(native(&arc_process, number), Ok(negated));

                    Ok(())
                },
            )
            .unwrap();
    });
}
