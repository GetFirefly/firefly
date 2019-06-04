use super::*;

mod with_big_integer_dividend;
mod with_small_integer_dividend;

#[test]
fn without_integer_dividend_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_integer(arc_process.clone()),
                    strategy::term::is_integer(arc_process.clone()),
                ),
                |(dividend, divisor)| {
                    prop_assert_eq!(
                        erlang::div_2(dividend, divisor, &arc_process),
                        Err(badarith!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_integer_dividend_without_integer_divisor_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_integer(arc_process.clone()),
                    strategy::term::is_not_integer(arc_process.clone()),
                ),
                |(dividend, divisor)| {
                    prop_assert_eq!(
                        erlang::div_2(dividend, divisor, &arc_process),
                        Err(badarith!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_integer_dividend_with_zero_divisor_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_integer(arc_process.clone()),
                    Just(0.into_process(&arc_process)),
                ),
                |(dividend, divisor)| {
                    prop_assert_eq!(
                        erlang::div_2(dividend, divisor, &arc_process),
                        Err(badarith!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}
