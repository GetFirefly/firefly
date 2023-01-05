use super::*;

#[test]
fn without_number_multiplicand_errors_badarith() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::float(),
                strategy::term::is_not_number(arc_process.clone()),
            )
        },
        |(arc_process, multiplier, multiplicand)| {
            prop_assert_badarith!(
                result(&arc_process, multiplier, multiplicand),
                format!(
                    "multiplier ({}) and multiplicand ({}) aren't both numbers",
                    multiplier, multiplicand
                )
            );

            Ok(())
        },
    );
}

#[test]
fn with_number_multiplicand_returns_float() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::float(),
                strategy::term::is_number(arc_process.clone()),
            )
        },
        |(arc_process, multiplier, multiplicand)| {
            let result = result(&arc_process, multiplier, multiplicand);

            prop_assert!(result.is_ok());

            let product = result.unwrap();

            prop_assert!(product.is_float());

            Ok(())
        },
    );
}

#[test]
fn with_float_multiplicand_without_underflow_or_overflow_returns_float() {
    with(|multiplier, process| {
        let multiplicand = 3.0.into();

        assert_eq!(
            result(process, multiplier, multiplicand),
            Ok(6.0.into())
        );
    })
}

#[test]
fn with_float_multiplicand_with_underflow_returns_min_float() {
    with(|multiplier, process| {
        let multiplicand = f64::MIN.into();

        assert_eq!(
            result(process, multiplier, multiplicand),
            Ok(f64::MIN.into())
        );
    })
}

#[test]
fn with_float_multiplicand_with_overflow_returns_max_float() {
    with(|multiplier, process| {
        let multiplicand = f64::MAX.into();

        assert_eq!(
            result(process, multiplier, multiplicand),
            Ok(f64::MAX.into())
        );
    })
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let multiplier = 2.0.into();

        f(multiplier, &process)
    })
}
