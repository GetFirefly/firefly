use super::*;

#[test]
fn with_atom_subtrahend_errors_badarith() {
    with_subtrahend_errors_badarith(|_| Term::str_to_atom("minuend", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_subtrahend_errors_badarith() {
    with_subtrahend_errors_badarith(|mut process| Term::local_reference(&mut process));
}

#[test]
fn with_empty_list_subtrahend_errors_badarith() {
    with_subtrahend_errors_badarith(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_subtrahend_errors_badarith() {
    with_subtrahend_errors_badarith(|mut process| {
        Term::cons(
            0.into_process(&mut process),
            1.into_process(&mut process),
            &mut process,
        )
    });
}

#[test]
fn with_small_integer_subtrahend_without_underflow_or_overflow_returns_small_integer() {
    with(|minuend, mut process| {
        let subtrahend = 3.into_process(&mut process);

        assert_eq!(
            erlang::subtract_2(minuend, subtrahend, &mut process),
            Ok((-1_isize).into_process(&mut process))
        );
    })
}

#[test]
fn with_small_integer_subtrahend_with_underflow_returns_big_integer() {
    with_process(|mut process| {
        let minuend = crate::integer::small::MIN.into_process(&mut process);
        let subtrahend = crate::integer::small::MAX.into_process(&mut process);

        assert_eq!(subtrahend.tag(), SmallInteger);

        let result = erlang::subtract_2(minuend, subtrahend, &mut process);

        assert!(result.is_ok());

        let difference = result.unwrap();

        assert_eq!(difference.tag(), Boxed);

        let unboxed_difference: &Term = difference.unbox_reference();

        assert_eq!(unboxed_difference.tag(), BigInteger);
    })
}

#[test]
fn with_small_integer_subtrahend_with_overflow_returns_big_integer() {
    with(|minuend, mut process| {
        let subtrahend = crate::integer::small::MIN.into_process(&mut process);

        assert_eq!(subtrahend.tag(), SmallInteger);

        let result = erlang::subtract_2(minuend, subtrahend, &mut process);

        assert!(result.is_ok());

        let difference = result.unwrap();

        assert_eq!(difference.tag(), Boxed);

        let unboxed_difference: &Term = difference.unbox_reference();

        assert_eq!(unboxed_difference.tag(), BigInteger);
    })
}

#[test]
fn with_big_integer_subtrahend_returns_big_integer() {
    with(|minuend, mut process| {
        let subtrahend = (crate::integer::small::MIN - 1).into_process(&mut process);

        assert_eq!(subtrahend.tag(), Boxed);

        let unboxed_subtrahend: &Term = subtrahend.unbox_reference();

        assert_eq!(unboxed_subtrahend.tag(), BigInteger);

        let result = erlang::subtract_2(minuend, subtrahend, &mut process);

        assert!(result.is_ok());

        let difference = result.unwrap();

        assert_eq!(difference.tag(), Boxed);

        let unboxed_difference: &Term = difference.unbox_reference();

        assert_eq!(unboxed_difference.tag(), BigInteger);
    })
}

#[test]
fn with_float_subtrahend_without_underflow_or_overflow_returns_float() {
    with(|minuend, mut process| {
        let subtrahend = 3.0.into_process(&mut process);

        assert_eq!(
            erlang::subtract_2(minuend, subtrahend, &mut process),
            Ok((-1.0).into_process(&mut process))
        );
    })
}

#[test]
fn with_float_subtrahend_with_underflow_returns_min_float() {
    with(|minuend, mut process| {
        let subtrahend = std::f64::MAX.into_process(&mut process);

        assert_eq!(
            erlang::subtract_2(minuend, subtrahend, &mut process),
            Ok(std::f64::MIN.into_process(&mut process))
        );
    })
}

#[test]
fn with_float_subtrahend_with_overflow_returns_max_float() {
    with(|minuend, mut process| {
        let subtrahend = std::f64::MIN.into_process(&mut process);

        assert_eq!(
            erlang::subtract_2(minuend, subtrahend, &mut process),
            Ok(std::f64::MAX.into_process(&mut process))
        );
    })
}

#[test]
fn with_local_pid_subtrahend_errors_badarith() {
    with_subtrahend_errors_badarith(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_subtrahend_errors_badarith() {
    with_subtrahend_errors_badarith(|mut process| {
        Term::external_pid(1, 2, 3, &mut process).unwrap()
    });
}

#[test]
fn with_tuple_subtrahend_errors_badarith() {
    with_subtrahend_errors_badarith(|mut process| Term::slice_to_tuple(&[], &mut process));
}

#[test]
fn with_map_is_subtrahend_errors_badarith() {
    with_subtrahend_errors_badarith(|mut process| Term::slice_to_map(&[], &mut process));
}

#[test]
fn with_heap_binary_subtrahend_errors_badarith() {
    with_subtrahend_errors_badarith(|mut process| Term::slice_to_binary(&[], &mut process));
}

#[test]
fn with_subbinary_subtrahend_errors_badarith() {
    with_subtrahend_errors_badarith(|mut process| {
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
        let minuend = 2.into_process(&mut process);

        f(minuend, &mut process)
    })
}

fn with_subtrahend_errors_badarith<M>(subtrahend: M)
where
    M: FnOnce(&mut Process) -> Term,
{
    super::errors_badarith(|mut process| {
        let minuend: Term = 2.into_process(&mut process);

        assert_eq!(minuend.tag(), SmallInteger);

        let subtrahend = subtrahend(&mut process);

        erlang::subtract_2(minuend, subtrahend, &mut process)
    });
}
