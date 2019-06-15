use super::*;

#[test]
fn with_float_minuend_with_integer_subtrahend_returns_float() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::float(arc_process.clone()),
                    strategy::term::is_integer(arc_process.clone()),
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

#[test]
fn with_float_minuend_with_float_subtrahend_returns_float() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::float(arc_process.clone()),
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

#[test]
fn with_float_subtrahend_with_underflow_returns_min_float() {
    with(|minuend, process| {
        let subtrahend = std::f64::MAX.into_process(&process);

        assert_eq!(
            erlang::subtract_2(minuend, subtrahend, &process),
            Ok(std::f64::MIN.into_process(&process))
        );
    })
}

#[test]
fn with_float_subtrahend_with_overflow_returns_max_float() {
    with(|minuend, process| {
        let subtrahend = std::f64::MIN.into_process(&process);

        assert_eq!(
            erlang::subtract_2(minuend, subtrahend, &process),
            Ok(std::f64::MAX.into_process(&process))
        );
    })
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let minuend = 2.0.into_process(&process);

        f(minuend, &process)
    })
}
