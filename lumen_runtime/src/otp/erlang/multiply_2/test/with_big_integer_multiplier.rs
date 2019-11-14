use super::*;

#[test]
fn without_number_multiplicand_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::integer::big(arc_process.clone()),
                    strategy::term::is_not_number(arc_process.clone()),
                ),
                |(multiplier, multiplicand)| {
                    prop_assert_eq!(
                        native(&arc_process, multiplier, multiplicand),
                        Err(badarith!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_integer_multiplicand_returns_big_integer() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::integer::big(arc_process.clone()),
                    strategy::term::is_integer(arc_process.clone()),
                ),
                |(multiplier, multiplicand)| {
                    let result = native(&arc_process, multiplier, multiplicand);

                    prop_assert!(result.is_ok());

                    let product = result.unwrap();

                    prop_assert!(product.is_boxed_bigint());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_float_multiplicand_without_underflow_or_overflow_returns_float() {
    with(|multiplier, process| {
        let multiplicand = process.float(3.0).unwrap();

        let result = native(process, multiplier, multiplicand);

        assert!(result.is_ok());

        let product = result.unwrap();

        assert!(product.is_boxed_float());
    })
}

#[test]
fn with_float_multiplicand_with_underflow_returns_min_float() {
    with(|multiplier, process| {
        let multiplicand = process.float(std::f64::MIN).unwrap();

        assert_eq!(
            native(process, multiplier, multiplicand),
            Ok(process.float(std::f64::MIN).unwrap())
        );
    })
}

#[test]
fn with_float_multiplicand_with_overflow_returns_max_float() {
    with(|multiplier, process| {
        let multiplicand = process.float(std::f64::MAX).unwrap();

        assert_eq!(
            native(process, multiplier, multiplicand),
            Ok(process.float(std::f64::MAX).unwrap())
        );
    })
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let multiplier: Term = process.integer(SmallInteger::MAX_VALUE + 1).unwrap();

        assert!(multiplier.is_boxed_bigint());

        f(multiplier, &process)
    })
}
