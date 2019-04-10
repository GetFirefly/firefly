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
fn with_small_integer_right_returns_big_integer() {
    with(|left, mut process| {
        let right: Term = 0b1010_1010_1010_1010_1010_1010_1010.into_process(&mut process);

        assert_eq!(right.tag(), SmallInteger);

        let result = erlang::bor_2(left, right, &mut process);

        assert!(result.is_ok());

        let output = result.unwrap();

        assert_eq!(output.tag(), Boxed);

        let unboxed_output: &Term = output.unbox_reference();

        assert_eq!(unboxed_output.tag(), BigInteger);
    });
}

#[test]
fn with_same_big_integer_right_returns_same_big_integer() {
    with(|left, mut process| {
        assert_eq!(erlang::bor_2(left, left, &mut process), Ok(left));
    })
}

#[test]
fn with_big_integer_right_returns_big_integer() {
    with(|left, mut process| {
        let right = <BigInt as Num>::from_str_radix(
            "1010".repeat(size_of::<usize>() * (8 / 4) * 2).as_ref(),
            2,
        )
        .unwrap()
        .into_process(&mut process);

        assert_eq!(right.tag(), Boxed);

        let unboxed_right: &Term = right.unbox_reference();

        assert_eq!(unboxed_right.tag(), BigInteger);

        let result = erlang::bor_2(left, right, &mut process);

        assert!(result.is_ok());

        let output = result.unwrap();

        println!("output = {:?}", output);

        assert_eq!(output.tag(), Boxed);

        let unboxed_output: &Term = output.unbox_reference();

        assert_eq!(unboxed_output.tag(), BigInteger);
        assert_eq!(
            output,
            <BigInt as Num>::from_str_radix(
                "1110".repeat(size_of::<usize>() * (8 / 4) * 2).as_ref(),
                2
            )
            .unwrap()
            .into_process(&mut process)
        );
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
        let left = <BigInt as Num>::from_str_radix(
            "1100".repeat(size_of::<usize>() * (8 / 4) * 2).as_ref(),
            2,
        )
        .unwrap()
        .into_process(&mut process);

        f(left, &mut process)
    })
}

fn with_right_errors_badarith<M>(right: M)
where
    M: FnOnce(&mut Process) -> Term,
{
    super::errors_badarith(|mut process| {
        let left = (crate::integer::small::MAX + 1).into_process(&mut process);

        assert_eq!(left.tag(), Boxed);

        let unboxed_left: &Term = left.unbox_reference();

        assert_eq!(unboxed_left.tag(), BigInteger);

        let right = right(&mut process);

        erlang::bor_2(left, right, &mut process)
    });
}
