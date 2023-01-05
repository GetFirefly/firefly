use super::*;

use crate::test::with_big_int;

#[test]
fn without_number_multiplicand_errors_badarith() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::integer::big(arc_process.clone()),
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
fn with_integer_multiplicand_returns_big_integer() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::integer::big(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
            )
        },
        |(arc_process, multiplier, multiplicand)| {
            let result = result(&arc_process, multiplier, multiplicand);

            prop_assert!(result.is_ok());

            let product = result.unwrap();

            prop_assert!(product.is_boxed_bigint());

            Ok(())
        },
    );
}

#[test]
fn with_float_multiplicand_without_underflow_or_overflow_returns_float() {
    with_big_int(|process, multiplier| {
        let multiplicand = 3.0.into();

        let result = result(process, multiplier, multiplicand);

        assert!(result.is_ok());

        let product = result.unwrap();

        assert!(product.is_float());
    })
}

#[test]
fn with_float_multiplicand_with_underflow_returns_min_float() {
    with_big_int(|process, multiplier| {
        let multiplicand = f64::MIN.into();

        assert_eq!(
            result(process, multiplier, multiplicand),
            Ok(f64::MIN.into())
        );
    })
}

#[test]
fn with_float_multiplicand_with_overflow_returns_max_float() {
    with_big_int(|process, multiplier| {
        let multiplicand = f64::MAX.into();

        assert_eq!(
            result(process, multiplier, multiplicand),
            Ok(f64::MAX.into())
        );
    })
}
