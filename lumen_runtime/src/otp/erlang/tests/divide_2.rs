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
                        Err(badarith!())
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
                        Err(badarith!())
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
                        Err(badarith!())
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

                    prop_assert_eq!(quotient.tag(), Boxed);

                    let unboxed_quotient: &Term = quotient.unbox_reference();

                    prop_assert_eq!(unboxed_quotient.tag(), Float);

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn number_is_not_zero(arc_process: Arc<Process>) -> impl Strategy<Value = Term> {
    strategy::term::is_number(arc_process).prop_filter("Number must not be zero", |number| {
        match number.tag() {
            SmallInteger => (unsafe { number.small_integer_to_isize() }) != 0,
            Boxed => {
                let unboxed: &Term = number.unbox_reference();

                match unboxed.tag() {
                    Float => {
                        let float: &Float = number.unbox_reference();

                        float.inner != 0.0
                    }
                    _ => true,
                }
            }
            _ => true,
        }
    })
}

fn zero(arc_process: Arc<Process>) -> impl Strategy<Value = Term> {
    prop_oneof![
        Just(0.into_process(&arc_process)),
        Just(0.0.into_process(&arc_process))
    ]
}
