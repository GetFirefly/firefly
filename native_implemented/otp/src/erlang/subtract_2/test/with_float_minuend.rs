use super::*;

#[test]
fn with_float_minuend_with_integer_subtrahend_returns_float() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::float(),
                strategy::term::is_integer(arc_process.clone()),
            )
        },
        |(arc_process, minuend, subtrahend)| {
            let result = result(&arc_process, minuend, subtrahend);

            prop_assert!(result.is_ok());

            let difference = result.unwrap();

            prop_assert!(difference.is_float());

            Ok(())
        },
    );
}

#[test]
fn with_float_minuend_with_float_subtrahend_returns_float() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::float(arc_process.clone()),
                strategy::term::float(arc_process.clone()),
            )
        },
        |(arc_process, minuend, subtrahend)| {
            let result = result(&arc_process, minuend, subtrahend);

            prop_assert!(result.is_ok());

            let difference = result.unwrap();

            prop_assert!(difference.is_float());

            Ok(())
        },
    );
}

#[test]
fn with_float_subtrahend_with_underflow_returns_min_float() {
    with(|minuend, process| {
        let subtrahend = f64::MAX.into();

        assert_eq!(
            result(&process, minuend, subtrahend),
            Ok(f64::MIN.into())
        );
    })
}

#[test]
fn with_float_subtrahend_with_overflow_returns_max_float() {
    with(|minuend, process| {
        let subtrahend = f64::MIN.into();

        assert_eq!(
            result(&process, minuend, subtrahend),
            Ok(f64::MAX.into())
        );
    })
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let minuend = 2.0.into();

        f(minuend, &process)
    })
}
