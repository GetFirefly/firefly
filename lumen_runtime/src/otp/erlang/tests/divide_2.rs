use super::*;

use proptest::prop_oneof;
use proptest::strategy::Strategy;

mod with_float_dividend;

#[test]
fn without_number_dividend_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_number(arc_process.clone()),
                    strategy::term::is_number(arc_process.clone()),
                ),
                |(dividend, divisor)| {
                    prop_assert_eq!(
                        erlang::divide_2(dividend, divisor, &arc_process),
                        Err(badarith!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_number_dividend_without_number_divisor_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_number(arc_process.clone()),
                    strategy::term::is_not_number(arc_process.clone()),
                ),
                |(dividend, divisor)| {
                    prop_assert_eq!(
                        erlang::divide_2(dividend, divisor, &arc_process),
                        Err(badarith!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_number_dividend_with_zero_divisor_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_number(arc_process.clone()),
                    zero(arc_process.clone()),
                ),
                |(dividend, divisor)| {
                    prop_assert_eq!(
                        erlang::divide_2(dividend, divisor, &arc_process),
                        Err(badarith!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_number_dividend_without_zero_number_divisor_returns_float() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_number(arc_process.clone()),
                    number_is_not_zero(arc_process.clone()),
                ),
                |(dividend, divisor)| {
                    let result = erlang::divide_2(dividend, divisor, &arc_process);

                    prop_assert!(result.is_ok());

                    let quotient = result.unwrap();

                    prop_assert!(quotient.is_float());

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn number_is_not_zero(arc_process: Arc<ProcessControlBlock>) -> BoxedStrategy<Term> {
    strategy::term::is_number(arc_process)
        .prop_filter("Number must not be zero", |number| {
            match number.to_typed_term().unwrap() {
                TypedTerm::SmallInteger(small_integer) => {
                    let i: isize = small_integer.into();

                    i != 0
                }
                TypedTerm::Boxed(unboxed) => match unboxed.to_typed_term().unwrap() {
                    TypedTerm::Float(float) => {
                        let f: f64 = float.into();

                        f != 0.0
                    }
                    _ => true,
                },
                _ => true,
            }
        })
        .boxed()
}

fn zero(arc_process: Arc<ProcessControlBlock>) -> BoxedStrategy<Term> {
    prop_oneof![
        Just(arc_process.integer(0)),
        Just(arc_process.float(0.0).unwrap())
    ]
    .boxed()
}
