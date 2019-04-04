use super::*;

#[test]
fn with_atom_divisor_errors_badarith() {
    with_divisor_errors_badarith(|_| Term::str_to_atom("dividend", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_divisor_errors_badarith() {
    with_divisor_errors_badarith(|mut process| Term::local_reference(&mut process));
}

#[test]
fn with_empty_list_divisor_errors_badarith() {
    with_divisor_errors_badarith(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_divisor_errors_badarith() {
    with_divisor_errors_badarith(|mut process| {
        Term::cons(
            0.into_process(&mut process),
            1.into_process(&mut process),
            &mut process,
        )
    });
}

#[test]
fn with_small_integer_zero_errors_badarith() {
    with(|dividend, mut process| {
        let divisor = 0.into_process(&mut process);

        assert_badarith!(erlang::divide_2(dividend, divisor, &mut process))
    })
}

#[test]
fn with_small_integer_divisor_returns_float() {
    with(|dividend, mut process| {
        let divisor = 4.into_process(&mut process);

        assert_eq!(
            erlang::divide_2(dividend, divisor, &mut process),
            Ok(0.5.into_process(&mut process))
        );
    })
}

#[test]
fn with_big_integer_divisor_returns_float() {
    with(|dividend, mut process| {
        let divisor = (crate::integer::small::MIN - 1).into_process(&mut process);

        assert_eq!(divisor.tag(), Boxed);

        let unboxed_divisor: &Term = divisor.unbox_reference();

        assert_eq!(unboxed_divisor.tag(), BigInteger);

        let result = erlang::divide_2(dividend, divisor, &mut process);

        assert!(result.is_ok());

        let quotient = result.unwrap();

        assert_eq!(quotient.tag(), Boxed);

        let unboxed_quotient: &Term = quotient.unbox_reference();

        assert_eq!(unboxed_quotient.tag(), Float);
    })
}

#[test]
fn with_float_zero_errors_badarith() {
    with(|dividend, mut process| {
        let divisor = 0.0.into_process(&mut process);

        assert_badarith!(erlang::divide_2(dividend, divisor, &mut process))
    })
}

#[test]
fn with_float_divisor_returns_float() {
    with(|dividend, mut process| {
        let divisor = 4.0.into_process(&mut process);

        assert_eq!(
            erlang::divide_2(dividend, divisor, &mut process),
            Ok(0.5.into_process(&mut process))
        );
    })
}

#[test]
fn with_local_pid_divisor_errors_badarith() {
    with_divisor_errors_badarith(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_divisor_errors_badarith() {
    with_divisor_errors_badarith(|mut process| Term::external_pid(1, 2, 3, &mut process).unwrap());
}

#[test]
fn with_tuple_divisor_errors_badarith() {
    with_divisor_errors_badarith(|mut process| Term::slice_to_tuple(&[], &mut process));
}

#[test]
fn with_map_is_divisor_errors_badarith() {
    with_divisor_errors_badarith(|mut process| Term::slice_to_map(&[], &mut process));
}

#[test]
fn with_heap_binary_divisor_errors_badarith() {
    with_divisor_errors_badarith(|mut process| Term::slice_to_binary(&[], &mut process));
}

#[test]
fn with_subbinary_divisor_errors_badarith() {
    with_divisor_errors_badarith(|mut process| {
        let original =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
        Term::subbinary(original, 0, 7, 2, 1, &mut process)
    });
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &mut Process) -> (),
{
    with_process(|mut process| {
        let dividend = 2.into_process(&mut process);

        f(dividend, &mut process)
    })
}

fn with_divisor_errors_badarith<M>(divisor: M)
where
    M: FnOnce(&mut Process) -> Term,
{
    super::errors_badarith(|mut process| {
        let dividend: Term = 2.into_process(&mut process);

        assert_eq!(dividend.tag(), SmallInteger);

        let divisor = divisor(&mut process);

        erlang::divide_2(dividend, divisor, &mut process)
    });
}
