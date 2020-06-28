use super::*;

#[test]
fn without_number_addend_errors_badarith() {
    super::without_number_addend_errors_badarith(file!(), strategy::term::integer::big);
}

#[test]
fn with_zero_small_integer_returns_same_big_integer() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::integer::big(arc_process.clone()),
            )
        },
        |(arc_process, augend)| {
            let addend = 0.into();

            prop_assert_eq!(result(&arc_process, augend, addend), Ok(augend));

            Ok(())
        },
    );
}

#[test]
fn that_is_positive_with_positive_small_integer_addend_returns_greater_big_integer() {
    that_is_positive_with_addend_returns_greater_big_integer(
        strategy::term::integer::small::positive,
    );
}

#[test]
fn that_is_positive_with_positive_big_integer_addend_returns_greater_big_integer() {
    that_is_positive_with_addend_returns_greater_big_integer(
        strategy::term::integer::big::positive,
    );
}

#[test]
fn with_float_addend_without_underflow_or_overflow_returns_float() {
    with(|augend, process| {
        let addend = process.float(3.0).unwrap();

        let result = result(&process, augend, addend);

        assert!(result.is_ok());

        let sum = result.unwrap();

        assert!(sum.is_boxed_float());
    })
}

#[test]
fn with_float_addend_with_underflow_returns_min_float() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::integer::big::negative(arc_process.clone()),
            )
        },
        |(arc_process, augend)| {
            let addend = arc_process.float(std::f64::MIN).unwrap();

            prop_assert_eq!(
                result(&arc_process, augend, addend),
                Ok(arc_process.float(std::f64::MIN).unwrap())
            );

            Ok(())
        },
    );
}

#[test]
fn with_float_addend_with_overflow_returns_max_float() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::integer::big::positive(arc_process.clone()),
            )
        },
        |(arc_process, augend)| {
            let addend = arc_process.float(std::f64::MAX).unwrap();

            prop_assert_eq!(
                result(&arc_process, augend, addend),
                Ok(arc_process.float(std::f64::MAX).unwrap())
            );

            Ok(())
        },
    );
}

fn that_is_positive_with_addend_returns_greater_big_integer<S, F>(addend_strategy: F)
where
    F: Fn(Arc<Process>) -> S,
    S: Strategy<Value = Term>,
{
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::integer::big::positive(arc_process.clone()),
                addend_strategy(arc_process),
            )
        },
        |(arc_process, augend, addend)| {
            let result = result(&arc_process, augend, addend);

            prop_assert!(result.is_ok());

            let sum = result.unwrap();

            prop_assert!(augend < sum);
            prop_assert!(addend < sum);
            prop_assert!(sum.is_boxed_bigint());

            Ok(())
        },
    );
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let augend = process.integer(SmallInteger::MAX_VALUE + 1).unwrap();

        assert!(augend.is_boxed_bigint());

        f(augend, &process)
    })
}
