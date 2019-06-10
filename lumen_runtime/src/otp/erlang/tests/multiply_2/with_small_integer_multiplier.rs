use super::*;

#[test]
fn without_number_multiplicand_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::integer::small(arc_process.clone()),
                    strategy::term::is_not_number(arc_process.clone()),
                ),
                |(multiplier, multiplicand)| {
                    prop_assert_eq!(
                        erlang::multiply_2(multiplier, multiplicand, &arc_process),
                        Err(badarith!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_small_integer_multiplicand_without_underflow_or_overflow_returns_small_integer() {
    with(|multiplier, process| {
        let multiplicand = 3.into_process(&process);

        assert_eq!(
            erlang::multiply_2(multiplier, multiplicand, &process),
            Ok(6.into_process(&process))
        );
    })
}

#[test]
fn with_small_integer_multiplicand_with_underflow_returns_big_integer() {
    with(|multiplier, process| {
        let multiplicand = crate::integer::small::MIN.into_process(&process);

        assert_eq!(multiplicand.tag(), SmallInteger);

        let result = erlang::multiply_2(multiplier, multiplicand, &process);

        assert!(result.is_ok());

        let product = result.unwrap();

        assert_eq!(product.tag(), Boxed);

        let unboxed_product: &Term = product.unbox_reference();

        assert_eq!(unboxed_product.tag(), BigInteger);
    })
}

#[test]
fn with_small_integer_multiplicand_with_overflow_returns_big_integer() {
    with(|multiplier, process| {
        let multiplicand = crate::integer::small::MAX.into_process(&process);

        assert_eq!(multiplicand.tag(), SmallInteger);

        let result = erlang::multiply_2(multiplier, multiplicand, &process);

        assert!(result.is_ok());

        let product = result.unwrap();

        assert_eq!(product.tag(), Boxed);

        let unboxed_product: &Term = product.unbox_reference();

        assert_eq!(unboxed_product.tag(), BigInteger);
    })
}

#[test]
fn with_big_integer_multiplicand_returns_big_integer() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::integer::small(arc_process.clone()),
                    strategy::term::integer::big(arc_process.clone()),
                ),
                |(multiplier, multiplicand)| {
                    let result = erlang::multiply_2(multiplier, multiplicand, &arc_process);

                    prop_assert!(result.is_ok());

                    let product = result.unwrap();

                    prop_assert_eq!(product.tag(), Boxed);

                    let unboxed_product: &Term = product.unbox_reference();

                    prop_assert_eq!(unboxed_product.tag(), BigInteger);

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_float_multiplicand_without_underflow_or_overflow_returns_float() {
    with(|multiplier, process| {
        let multiplicand = 3.0.into_process(&process);

        assert_eq!(
            erlang::multiply_2(multiplier, multiplicand, &process),
            Ok(6.0.into_process(&process))
        );
    })
}

#[test]
fn with_float_multiplicand_with_underflow_returns_min_float() {
    with(|multiplier, process| {
        let multiplicand = std::f64::MIN.into_process(&process);

        assert_eq!(
            erlang::multiply_2(multiplier, multiplicand, &process),
            Ok(std::f64::MIN.into_process(&process))
        );
    })
}

#[test]
fn with_float_multiplicand_with_overflow_returns_max_float() {
    with(|multiplier, process| {
        let multiplicand = std::f64::MAX.into_process(&process);

        assert_eq!(
            erlang::multiply_2(multiplier, multiplicand, &process),
            Ok(std::f64::MAX.into_process(&process))
        );
    })
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let multiplier = 2.into_process(&process);

        f(multiplier, &process)
    })
}
