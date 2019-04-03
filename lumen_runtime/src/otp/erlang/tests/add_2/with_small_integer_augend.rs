use super::*;

#[test]
fn with_atom_addend_errors_badarith() {
    with_addend_errors_badarith(|_| Term::str_to_atom("augend", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_addend_errors_badarith() {
    with_addend_errors_badarith(|mut process| Term::local_reference(&mut process));
}

#[test]
fn with_empty_list_addend_errors_badarith() {
    with_addend_errors_badarith(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_addend_errors_badarith() {
    with_addend_errors_badarith(|mut process| {
        Term::cons(
            0.into_process(&mut process),
            1.into_process(&mut process),
            &mut process,
        )
    });
}

#[test]
fn with_small_integer_addend_without_underflow_or_overflow_returns_small_integer() {
    with(|augend, mut process| {
        let addend = 3.into_process(&mut process);

        assert_eq!(
            erlang::add_2(augend, addend, &mut process),
            Ok(5.into_process(&mut process))
        );
    })
}

#[test]
fn with_small_integer_addend_with_underflow_returns_big_integer() {
    with_process(|mut process| {
        let augend = (-1_isize).into_process(&mut process);
        let addend = crate::integer::small::MIN.into_process(&mut process);

        assert_eq!(addend.tag(), SmallInteger);

        let result = erlang::add_2(augend, addend, &mut process);

        assert!(result.is_ok());

        let sum = result.unwrap();

        assert_eq!(sum.tag(), Boxed);

        let unboxed_sum: &Term = sum.unbox_reference();

        assert_eq!(unboxed_sum.tag(), BigInteger);
    })
}

#[test]
fn with_small_integer_addend_with_overflow_returns_big_integer() {
    with(|augend, mut process| {
        let addend = crate::integer::small::MAX.into_process(&mut process);

        assert_eq!(addend.tag(), SmallInteger);

        let result = erlang::add_2(augend, addend, &mut process);

        assert!(result.is_ok());

        let sum = result.unwrap();

        assert_eq!(sum.tag(), Boxed);

        let unboxed_sum: &Term = sum.unbox_reference();

        assert_eq!(unboxed_sum.tag(), BigInteger);
    })
}

#[test]
fn with_big_integer_addend_returns_big_integer() {
    with(|augend, mut process| {
        let addend = (crate::integer::small::MAX + 1).into_process(&mut process);

        assert_eq!(addend.tag(), Boxed);

        let unboxed_addend: &Term = addend.unbox_reference();

        assert_eq!(unboxed_addend.tag(), BigInteger);

        let result = erlang::add_2(augend, addend, &mut process);

        assert!(result.is_ok());

        let sum = result.unwrap();

        assert_eq!(sum.tag(), Boxed);

        let unboxed_sum: &Term = sum.unbox_reference();

        assert_eq!(unboxed_sum.tag(), BigInteger);
    })
}

#[test]
fn with_float_addend_without_underflow_or_overflow_returns_float() {
    with(|augend, mut process| {
        let addend = 3.0.into_process(&mut process);

        assert_eq!(
            erlang::add_2(augend, addend, &mut process),
            Ok(5.0.into_process(&mut process))
        );
    })
}

#[test]
fn with_float_addend_with_underflow_returns_min_float() {
    with(|augend, mut process| {
        let addend = std::f64::MIN.into_process(&mut process);

        assert_eq!(
            erlang::add_2(augend, addend, &mut process),
            Ok(std::f64::MIN.into_process(&mut process))
        );
    })
}

#[test]
fn with_float_addend_with_overflow_returns_max_float() {
    with(|augend, mut process| {
        let addend = std::f64::MAX.into_process(&mut process);

        assert_eq!(
            erlang::add_2(augend, addend, &mut process),
            Ok(std::f64::MAX.into_process(&mut process))
        );
    })
}

#[test]
fn with_local_pid_addend_errors_badarith() {
    with_addend_errors_badarith(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_addend_errors_badarith() {
    with_addend_errors_badarith(|mut process| Term::external_pid(1, 2, 3, &mut process).unwrap());
}

#[test]
fn with_tuple_addend_errors_badarith() {
    with_addend_errors_badarith(|mut process| Term::slice_to_tuple(&[], &mut process));
}

#[test]
fn with_map_is_addend_errors_badarith() {
    with_addend_errors_badarith(|mut process| Term::slice_to_map(&[], &mut process));
}

#[test]
fn with_heap_binary_addend_errors_badarith() {
    with_addend_errors_badarith(|mut process| Term::slice_to_binary(&[], &mut process));
}

#[test]
fn with_subbinary_addend_errors_badarith() {
    with_addend_errors_badarith(|mut process| {
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
        let augend = 2.into_process(&mut process);

        f(augend, &mut process)
    })
}

fn with_addend_errors_badarith<M>(addend: M)
where
    M: FnOnce(&mut Process) -> Term,
{
    super::errors_badarith(|mut process| {
        let augend: Term = 2.into_process(&mut process);

        assert_eq!(augend.tag(), SmallInteger);

        let addend = addend(&mut process);

        erlang::add_2(augend, addend, &mut process)
    });
}
