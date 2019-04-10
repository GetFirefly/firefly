use super::*;

use std::mem::size_of;

use num_traits::Num;

#[test]
fn with_atom_right_errors_badarith() {
    with_right_errors_badarith(|_| Term::str_to_atom("right", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_right_errors_badarith() {
    with_right_errors_badarith(|mut process| Term::local_reference(&mut process));
}

#[test]
fn with_empty_list_right_errors_badarith() {
    with_right_errors_badarith(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_right_errors_badarith() {
    with_right_errors_badarith(|mut process| {
        Term::cons(
            0.into_process(&mut process),
            1.into_process(&mut process),
            &mut process,
        )
    });
}

#[test]
fn with_same_small_integer_right_returns_zero() {
    with(|left, mut process| {
        assert_eq!(
            erlang::bxor_2(left, left, &mut process),
            Ok(0.into_process(&mut process))
        );
    })
}

#[test]
fn with_small_integer_right_returns_small_integer() {
    with_process(|mut process| {
        // all combinations of `0` and `1` bit.
        let left = 0b1100.into_process(&mut process);
        let right = 0b1010.into_process(&mut process);

        assert_eq!(
            erlang::bxor_2(left, right, &mut process),
            Ok(0b0110.into_process(&mut process))
        );
    })
}

#[test]
fn with_big_integer_right_returns_big_integer() {
    with_process(|mut process| {
        let left: Term = 0b1100_1100_1100_1100_1100_1100_1100_isize.into_process(&mut process);

        assert_eq!(left.tag(), SmallInteger);

        let right = <BigInt as Num>::from_str_radix(
            "1010".repeat(size_of::<usize>() * (8 / 4) * 2).as_ref(),
            2,
        )
        .unwrap()
        .into_process(&mut process);

        assert_eq!(right.tag(), Boxed);

        let unboxed_right: &Term = right.unbox_reference();

        assert_eq!(unboxed_right.tag(), BigInteger);

        let result = erlang::bxor_2(left, right, &mut process);

        assert!(result.is_ok());

        let output = result.unwrap();

        assert_eq!(output.tag(), Boxed);

        let unboxed_output: &Term = output.unbox_reference();

        assert_eq!(unboxed_output.tag(), BigInteger);
    })
}

#[test]
fn with_float_right_errors_badarith() {
    with_right_errors_badarith(|mut process| 1.0.into_process(&mut process));
}

#[test]
fn with_local_pid_right_errors_badarith() {
    with_right_errors_badarith(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_right_errors_badarith() {
    with_right_errors_badarith(|mut process| Term::external_pid(1, 2, 3, &mut process).unwrap());
}

#[test]
fn with_tuple_right_errors_badarith() {
    with_right_errors_badarith(|mut process| Term::slice_to_tuple(&[], &mut process));
}

#[test]
fn with_map_is_right_errors_badarith() {
    with_right_errors_badarith(|mut process| Term::slice_to_map(&[], &mut process));
}

#[test]
fn with_heap_binary_right_errors_badarith() {
    with_right_errors_badarith(|mut process| Term::slice_to_binary(&[], &mut process));
}

#[test]
fn with_subbinary_right_errors_badarith() {
    with_right_errors_badarith(|mut process| bitstring!(1 :: 1, &mut process));
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &mut Process) -> (),
{
    with_process(|mut process| {
        let left = 2.into_process(&mut process);

        f(left, &mut process)
    })
}

fn with_right_errors_badarith<M>(right: M)
where
    M: FnOnce(&mut Process) -> Term,
{
    super::errors_badarith(|mut process| {
        let left: Term = 2.into_process(&mut process);

        assert_eq!(left.tag(), SmallInteger);

        let right = right(&mut process);

        erlang::bxor_2(left, right, &mut process)
    });
}
