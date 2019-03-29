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
    let heap_binary_term = Term::slice_to_binary("-576460752303423488".as_bytes(), &mut process);

    let integer_result = erlang::binary_to_integer_1(heap_binary_term, &mut process);

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
    let heap_binary_term = Term::slice_to_binary("576460752303423487".as_bytes(), &mut process);

    let integer_result = erlang::binary_to_integer_1(heap_binary_term, &mut process);

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
    let heap_binary_term = Term::slice_to_binary("-576460752303423489".as_bytes(), &mut process);

    let integer_result = erlang::binary_to_integer_1(heap_binary_term, &mut process);

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
    let heap_binary_term = Term::slice_to_binary("576460752303423488".as_bytes(), &mut process);

    let integer_result = erlang::binary_to_integer_1(heap_binary_term, &mut process);

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
fn with_heap_binary_with_non_decimal_errors_badarg() {
    errors_badarg(|mut process| Term::slice_to_binary("FF".as_bytes(), &mut process));
}

#[test]
fn with_subbinary_with_min_small_integer_returns_small_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // <<1::1, "-576460752303423488">>
    let heap_binary_term = Term::slice_to_binary(
        &[
            150,
            154,
            155,
            155,
            26,
            27,
            24,
            27,
            154,
            153,
            25,
            152,
            25,
            154,
            25,
            25,
            154,
            28,
            28,
            0b0000_0000,
        ],
        &mut process,
    );
    let subbinary_term = Term::subbinary(heap_binary_term, 0, 1, 19, 0, &mut process);

    let integer_result = erlang::binary_to_integer_1(subbinary_term, &mut process);

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
    // <<1::1, "576460752303423487">>
    let heap_binary_term = Term::slice_to_binary(
        &[
            154,
            155,
            155,
            26,
            27,
            24,
            27,
            154,
            153,
            25,
            152,
            25,
            154,
            25,
            25,
            154,
            28,
            27,
            0b1000_0000,
        ],
        &mut process,
    );
    let subbinary_term = Term::subbinary(heap_binary_term, 0, 1, 18, 0, &mut process);

    let integer_result = erlang::binary_to_integer_1(subbinary_term, &mut process);

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
    // <<1::1, "-576460752303423489">>
    let heap_binary_term = Term::slice_to_binary(
        &[
            150,
            154,
            155,
            155,
            26,
            27,
            24,
            27,
            154,
            153,
            25,
            152,
            25,
            154,
            25,
            25,
            154,
            28,
            28,
            0b1000_0000,
        ],
        &mut process,
    );
    let subbinary_term = Term::subbinary(heap_binary_term, 0, 1, 19, 0, &mut process);

    let integer_result = erlang::binary_to_integer_1(subbinary_term, &mut process);

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
fn with_subbinary_with_greater_than_max_small_integer_returns_big_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // <<1::1, "576460752303423488">>
    let heap_binary_term = Term::slice_to_binary(
        &[
            154,
            155,
            155,
            26,
            27,
            24,
            27,
            154,
            153,
            25,
            152,
            25,
            154,
            25,
            25,
            154,
            28,
            28,
            0b0000_0000,
        ],
        &mut process,
    );
    let subbinary_term = Term::subbinary(heap_binary_term, 0, 1, 18, 0, &mut process);

    let integer_result = erlang::binary_to_integer_1(subbinary_term, &mut process);

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
fn with_subbinary_with_non_decimal_errors_badarg() {
    errors_badarg(|mut process| {
        // <<1:1, "FF>>
        let original = Term::slice_to_binary(&[163, 35, 0b000_0000], &mut process);
        Term::subbinary(original, 0, 1, 2, 0, &mut process)
    });
}

fn errors_badarg<F>(binary: F)
where
    F: FnOnce(&mut Process) -> Term,
{
    super::errors_badarg(|mut process| {
        erlang::binary_to_integer_1(binary(&mut process), &mut process)
    });
}
