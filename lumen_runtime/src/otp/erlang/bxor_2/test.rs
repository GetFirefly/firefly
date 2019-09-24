mod with_big_integer_left;
mod with_small_integer_left;

use num_bigint::BigInt;

use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarith;

use crate::otp::erlang::bxor_2::native;
use crate::scheduler::{with_process, with_process_arc};
use crate::test::strategy;

#[test]
fn without_integer_left_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_integer(arc_process.clone()),
                    strategy::term::is_integer(arc_process.clone()),
                ),
                |(left, right)| {
                    prop_assert_eq!(native(&arc_process, left, right), Err(badarith!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn without_integer_left_without_integer_right_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_integer(arc_process.clone()),
                    strategy::term::is_not_integer(arc_process.clone()),
                ),
                |(left, right)| {
                    prop_assert_eq!(native(&arc_process, left, right), Err(badarith!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_integer_returns_zero() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_integer(arc_process.clone()),
                |operand| {
                    prop_assert_eq!(
                        native(&arc_process, operand, operand),
                        Ok(arc_process.integer(0).unwrap())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
