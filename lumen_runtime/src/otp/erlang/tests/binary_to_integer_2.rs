use super::*;

use std::sync::{Arc, RwLock};

use num_traits::Num;

use crate::environment::{self, Environment};
use crate::process::IntoProcess;

#[test]
fn with_atom_errors_badarg() {
    errors_badarg(|_| Term::str_to_atom("😈🤘", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_errors_badarg() {
    errors_badarg(|mut process| Term::local_reference(&mut process));
}

#[test]
fn with_empty_list_errors_badarg() {
    errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_errors_badarg() {
    errors_badarg(|mut process| list_term(&mut process));
}

#[test]
fn with_small_integer_errors_badarg() {
    errors_badarg(|mut process| 0usize.into_process(&mut process));
}

#[test]
fn with_big_integer_errors_badarg() {
    errors_badarg(|mut process| {
        <BigInt as Num>::from_str_radix("18446744073709551616", 10)
            .unwrap()
            .into_process(&mut process)
    });
}

#[test]
fn with_float_errors_badarg() {
    errors_badarg(|mut process| 1.0.into_process(&mut process));
}

#[test]
fn with_local_pid_errors_badarg() {
    errors_badarg(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_errors_badarg() {
    errors_badarg(|mut process| Term::external_pid(1, 0, 0, &mut process).unwrap());
}

#[test]
fn with_tuple_errors_badarg() {
    errors_badarg(|mut process| Term::slice_to_tuple(&[], &mut process));
}

#[test]
fn with_map_errors_badarg() {
    errors_badarg(|mut process| Term::slice_to_map(&[], &mut process));
}

#[test]
fn with_heap_binary_with_min_small_integer_returns_small_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary = Term::slice_to_binary("-800000000000000".as_bytes(), &mut process);
    let base: Term = 16.into_process(&mut process);

    let integer_result = erlang::binary_to_integer_2(binary, base, &mut process);

    assert_eq!(
        integer_result,
        Ok(<BigInt as Num>::from_str_radix("-576460752303423488", 10)
            .unwrap()
            .into_process(&mut process))
    );
    assert_eq!(integer_result.unwrap().tag(), SmallInteger);
}

#[test]
fn with_heap_binary_with_max_small_integer_returns_small_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary = Term::slice_to_binary("7FFFFFFFFFFFFFF".as_bytes(), &mut process);
    let base: Term = 16.into_process(&mut process);

    let integer_result = erlang::binary_to_integer_2(binary, base, &mut process);

    assert_eq!(
        integer_result,
        Ok(<BigInt as Num>::from_str_radix("576460752303423487", 10)
            .unwrap()
            .into_process(&mut process))
    );
    assert_eq!(integer_result.unwrap().tag(), SmallInteger);
}

#[test]
fn with_heap_binary_with_less_than_min_small_integer_returns_big_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary = Term::slice_to_binary("-800000000000001".as_bytes(), &mut process);
    let base: Term = 16.into_process(&mut process);

    let integer_result = erlang::binary_to_integer_2(binary, base, &mut process);

    assert_eq!(
        integer_result,
        Ok(<BigInt as Num>::from_str_radix("-576460752303423489", 10)
            .unwrap()
            .into_process(&mut process))
    );

    let integer = integer_result.unwrap();

    assert_eq!(integer.tag(), Boxed);

    let unboxed: &Term = integer.unbox_reference();

    assert_eq!(unboxed.tag(), BigInteger);
}

#[test]
fn with_heap_binary_with_greater_than_max_small_integer_returns_big_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary = Term::slice_to_binary("800000000000000".as_bytes(), &mut process);
    let base: Term = 16.into_process(&mut process);

    let integer_result = erlang::binary_to_integer_2(binary, base, &mut process);

    assert_eq!(
        integer_result,
        Ok(<BigInt as Num>::from_str_radix("576460752303423488", 10)
            .unwrap()
            .into_process(&mut process))
    );

    let integer = integer_result.unwrap();

    assert_eq!(integer.tag(), Boxed);

    let unboxed: &Term = integer.unbox_reference();

    assert_eq!(unboxed.tag(), BigInteger);
}

#[test]
fn with_subbinary_with_min_small_integer_returns_small_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // <<1::1, Integer.to_string(-576460752303423488, 16) :: binary>>
    let original = Term::slice_to_binary(
        &[
            150,
            156,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            0b0000_0000,
        ],
        &mut process,
    );
    let binary = Term::subbinary(original, 0, 1, 16, 0, &mut process);
    let base: Term = 16.into_process(&mut process);

    let integer_result = erlang::binary_to_integer_2(binary, base, &mut process);

    assert_eq!(
        integer_result,
        Ok(<BigInt as Num>::from_str_radix("-576460752303423488", 10)
            .unwrap()
            .into_process(&mut process))
    );
    assert_eq!(integer_result.unwrap().tag(), SmallInteger);
}

#[test]
fn with_subbinary_with_max_small_integer_returns_small_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // <<1::1, Integer.to_string(576460752303423487, 16) :: binary>>
    let original = Term::slice_to_binary(
        &[
            155,
            163,
            35,
            35,
            35,
            35,
            35,
            35,
            35,
            35,
            35,
            35,
            35,
            35,
            35,
            0b0000_0000,
        ],
        &mut process,
    );
    let binary = Term::subbinary(original, 0, 1, 15, 0, &mut process);
    let base: Term = 16.into_process(&mut process);

    let integer_result = erlang::binary_to_integer_2(binary, base, &mut process);

    assert_eq!(
        integer_result,
        Ok(<BigInt as Num>::from_str_radix("576460752303423487", 10)
            .unwrap()
            .into_process(&mut process))
    );
    assert_eq!(integer_result.unwrap().tag(), SmallInteger);
}

#[test]
fn with_subbinary_with_less_than_min_small_integer_returns_big_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // <<1::1, Integer.to_string(-576460752303423489, 16) :: binary>>
    let original = Term::slice_to_binary(
        &[
            150,
            156,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            0b1000_0000,
        ],
        &mut process,
    );
    let binary = Term::subbinary(original, 0, 1, 16, 0, &mut process);
    let base: Term = 16.into_process(&mut process);

    let integer_result = erlang::binary_to_integer_2(binary, base, &mut process);

    assert_eq!(
        integer_result,
        Ok(
            <BigInt as Num>::from_str_radix("-5764_60_752_303_423_489", 10)
                .unwrap()
                .into_process(&mut process)
        )
    );

    let integer = integer_result.unwrap();

    assert_eq!(integer.tag(), Boxed);

    let unboxed: &Term = integer.unbox_reference();

    assert_eq!(unboxed.tag(), BigInteger);
}

#[test]
fn with_subbinary_with_greater_than_max_small_integer_returns_big_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // <<1::1, Integer.to_string(576460752303423488, 16) :: binary>>
    let original = Term::slice_to_binary(
        &[
            156,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            0b0000_0000,
        ],
        &mut process,
    );
    let binary = Term::subbinary(original, 0, 1, 15, 0, &mut process);
    let base: Term = 16.into_process(&mut process);

    let integer_result = erlang::binary_to_integer_2(binary, base, &mut process);

    assert_eq!(
        integer_result,
        Ok(<BigInt as Num>::from_str_radix("576460752303423488", 10)
            .unwrap()
            .into_process(&mut process))
    );

    let integer = integer_result.unwrap();

    assert_eq!(integer.tag(), Boxed);

    let unboxed: &Term = integer.unbox_reference();

    assert_eq!(unboxed.tag(), BigInteger);
}

fn errors_badarg<F>(binary: F)
where
    F: FnOnce(&mut Process) -> Term,
{
    super::errors_badarg(|mut process| {
        let base: Term = 16.into_process(&mut process);

        erlang::binary_to_integer_2(binary(&mut process), base, &mut process)
    });
}
