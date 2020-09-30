use std::sync::Arc;

use proptest::prop_assert;
use proptest::strategy::{BoxedStrategy, Just, Strategy};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::divide_2::result;
use crate::test::strategy;

// FIXME https://github.com/lumen/lumen/issues/650 and then remove for integration test
#[test]
fn with_number_dividend_without_zero_number_divisor_returns_float() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_number(arc_process.clone()),
                number_is_not_zero(arc_process.clone()),
            )
        },
        |(arc_process, dividend, divisor)| {
            let result = result(&arc_process, dividend, divisor);

            prop_assert!(result.is_ok());

            let quotient = result.unwrap();

            prop_assert!(quotient.is_float());

            Ok(())
        },
    );
}

fn number_is_not_zero(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    strategy::term::is_number(arc_process)
        .prop_filter("Number must not be zero", |number| {
            match number.decode().unwrap() {
                TypedTerm::SmallInteger(small_integer) => {
                    let i: isize = small_integer.into();

                    i != 0
                }
                TypedTerm::Float(float) => {
                    let f: f64 = float.into();

                    f != 0.0
                }
                _ => true,
            }
        })
        .boxed()
}
