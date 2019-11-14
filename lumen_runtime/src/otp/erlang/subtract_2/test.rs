mod with_float_minuend;
mod with_integer_minuend;

use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq};

use liblumen_alloc::badarith;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::subtract_2::native;
use crate::scheduler::{with_process, with_process_arc};
use crate::test::strategy;

#[test]
fn without_number_minuend_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_number(arc_process.clone()),
                    strategy::term::is_number(arc_process.clone()),
                ),
                |(minuend, subtrahend)| {
                    prop_assert_eq!(
                        native(&arc_process, minuend, subtrahend),
                        Err(badarith!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_number_minuend_without_number_subtrahend_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_number(arc_process.clone()),
                    strategy::term::is_not_number(arc_process.clone()),
                ),
                |(minuend, subtrahend)| {
                    prop_assert_eq!(
                        native(&arc_process, minuend, subtrahend),
                        Err(badarith!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_integer_minuend_with_integer_subtrahend_returns_integer() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_integer(arc_process.clone()),
                    strategy::term::is_integer(arc_process.clone()),
                ),
                |(minuend, subtrahend)| {
                    let result = native(&arc_process, minuend, subtrahend);

                    prop_assert!(result.is_ok());

                    let difference = result.unwrap();

                    prop_assert!(difference.is_integer());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_integer_minuend_with_float_subtrahend_returns_float() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_integer(arc_process.clone()),
                    strategy::term::float(arc_process.clone()),
                ),
                |(minuend, subtrahend)| {
                    let result = native(&arc_process, minuend, subtrahend);

                    prop_assert!(result.is_ok());

                    let difference = result.unwrap();

                    prop_assert!(difference.is_float());

                    Ok(())
                },
            )
            .unwrap();
    });
}
