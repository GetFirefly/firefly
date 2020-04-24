use super::*;

#[test]
fn without_number_addend_errors_badarith() {
    super::without_number_addend_errors_badarith(file!(), strategy::term::integer::small);
}

#[test]
fn with_small_integer_addend_without_underflow_or_overflow_returns_small_integer() {
    with(|augend, process| {
        let addend = process.integer(3).unwrap();

        assert_eq!(result(&process, augend, addend), Ok(5.into()));
    })
}

#[test]
fn with_small_integer_addend_with_underflow_returns_big_integer() {
    with_process(|process| {
        let augend = process.integer(-1_isize).unwrap();
        let addend = process.integer(SmallInteger::MIN_VALUE).unwrap();

        assert!(addend.is_smallint());

        let result = result(&process, augend, addend);

        assert!(result.is_ok());

        let sum = result.unwrap();

        assert!(sum.is_boxed_bigint());
    })
}

#[test]
fn with_small_integer_addend_with_overflow_returns_big_integer() {
    with(|augend, process| {
        let addend = process.integer(SmallInteger::MAX_VALUE).unwrap();

        assert!(addend.is_smallint());

        let result = result(&process, augend, addend);

        assert!(result.is_ok());

        let sum = result.unwrap();

        assert!(sum.is_boxed_bigint());
    })
}

#[test]
fn with_big_integer_addend_returns_big_integer() {
    with(|augend, process| {
        let addend = process.integer(SmallInteger::MAX_VALUE + 1).unwrap();

        assert!(addend.is_boxed_bigint());

        let result = result(&process, augend, addend);

        assert!(result.is_ok());

        let sum = result.unwrap();

        assert!(sum.is_boxed_bigint());
    })
}

#[test]
fn with_float_addend_without_underflow_or_overflow_returns_float() {
    with(|augend, process| {
        let addend = process.float(3.0).unwrap();

        assert_eq!(
            result(&process, augend, addend),
            Ok(process.float(5.0).unwrap())
        );
    })
}

#[test]
fn with_float_addend_with_underflow_returns_min_float() {
    with(|augend, process| {
        let addend = process.float(std::f64::MIN).unwrap();

        assert_eq!(
            result(&process, augend, addend),
            Ok(process.float(std::f64::MIN).unwrap())
        );
    })
}

#[test]
fn with_float_addend_with_overflow_returns_max_float() {
    with(|augend, process| {
        let addend = process.float(std::f64::MAX).unwrap();

        assert_eq!(
            result(&process, augend, addend),
            Ok(process.float(std::f64::MAX).unwrap())
        );
    })
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let augend = process.integer(2).unwrap();

        f(augend, &process)
    })
}
