mod with_big_integer_integer;
mod with_small_integer_integer;

use num_bigint::BigInt;

use proptest::prop_assert_eq;
use proptest::strategy::{BoxedStrategy, Strategy};
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarith;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Encoded;

use crate::otp::erlang;
use crate::otp::erlang::bsr_2::native;
use crate::scheduler::{with_process, with_process_arc};
use crate::test::strategy;

#[test]
fn without_integer_integer_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_integer(arc_process.clone()),
                    strategy::term::is_integer(arc_process.clone()),
                ),
                |(integer, shift)| {
                    prop_assert_eq!(
                        native(&arc_process, integer, shift),
                        Err(badarith!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_integer_integer_without_integer_shift_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_integer(arc_process.clone()),
                    strategy::term::is_not_integer(arc_process.clone()),
                ),
                |(integer, shift)| {
                    prop_assert_eq!(
                        native(&arc_process, integer, shift),
                        Err(badarith!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_integer_integer_with_zero_shift_returns_same_integer() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_integer(arc_process.clone()),
                |integer| {
                    let shift = arc_process.integer(0).unwrap();

                    prop_assert_eq!(native(&arc_process, integer, shift), Ok(integer));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_integer_integer_with_integer_shift_is_the_same_as_bsl_with_negated_shift() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(strategy::term::is_integer(arc_process.clone()), shift()),
                |(integer, shift)| {
                    let negated_shift = -1 * shift;

                    prop_assert_eq!(
                        native(
                            &arc_process,
                            integer,
                            arc_process.integer(shift as isize).unwrap(),
                        ),
                        erlang::bsl_2::native(
                            &arc_process,
                            integer,
                            arc_process.integer(negated_shift as isize).unwrap(),
                        )
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn shift() -> BoxedStrategy<i8> {
    // any::<i8> is not symmetric because i8::MIN is -128 while i8::MAX is 127, so make symmetric
    // range
    (-127_i8..=127_i8).boxed()
}
