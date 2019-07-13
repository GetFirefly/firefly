use super::*;

mod with_float_minuend;
mod with_integer_minuend;

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
                        erlang::subtract_2(minuend, subtrahend, &arc_process),
                        Err(badarith!().into())
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
                        erlang::subtract_2(minuend, subtrahend, &arc_process),
                        Err(badarith!().into())
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
                    let result = erlang::subtract_2(minuend, subtrahend, &arc_process);

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
                    let result = erlang::subtract_2(minuend, subtrahend, &arc_process);

                    prop_assert!(result.is_ok());

                    let difference = result.unwrap();

                    prop_assert!(difference.is_float());

                    Ok(())
                },
            )
            .unwrap();
    });
}
