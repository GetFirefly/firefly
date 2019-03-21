use super::*;

use std::sync::{Arc, RwLock};

use num_traits::Num;

use crate::environment::{self, Environment};
use crate::process::IntoProcess;

#[test]
fn with_atom_returns_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let atom_term = Term::str_to_atom("😈🤘", Existence::DoNotCare, &mut process).unwrap();
    let base_term: Term = 16.into_process(&mut process);

    assert_bad_argument!(
        erlang::binary_to_integer_2(atom_term, base_term, &mut process),
        &mut process
    );
}

#[test]
fn with_empty_list_returns_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let base_term: Term = 16.into_process(&mut process);

    assert_bad_argument!(
        erlang::binary_to_integer_2(Term::EMPTY_LIST, base_term, &mut process),
        &mut process
    );
}

#[test]
fn with_list_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list_term = list_term(&mut process);
    let base_term: Term = 16.into_process(&mut process);

    assert_bad_argument!(
        erlang::binary_to_integer_2(list_term, base_term, &mut process),
        &mut process
    );
}

#[test]
fn with_small_integer_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let small_integer_term = 0usize.into_process(&mut process);
    let base_term: Term = 16.into_process(&mut process);

    assert_bad_argument!(
        erlang::binary_to_integer_2(small_integer_term, base_term, &mut process),
        &mut process
    );
}

#[test]
fn with_big_integer_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let big_integer_term: Term = <BigInt as Num>::from_str_radix("18446744073709551616", 10)
        .unwrap()
        .into_process(&mut process);
    let base_term: Term = 16.into_process(&mut process);

    assert_bad_argument!(
        erlang::binary_to_integer_2(big_integer_term, base_term, &mut process),
        &mut process
    );
}

#[test]
fn with_float_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let float_term = 1.0.into_process(&mut process);
    let base_term: Term = 16.into_process(&mut process);

    assert_bad_argument!(
        erlang::binary_to_integer_2(float_term, base_term, &mut process),
        &mut process
    );
}

#[test]
fn with_local_pid_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let local_pid_term = Term::local_pid(0, 0, &mut process).unwrap();
    let base_term: Term = 16.into_process(&mut process);

    assert_bad_argument!(
        erlang::binary_to_integer_2(local_pid_term, base_term, &mut process),
        &mut process
    );
}

#[test]
fn with_external_pid_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();
    let base_term: Term = 16.into_process(&mut process);

    assert_bad_argument!(
        erlang::binary_to_integer_2(external_pid_term, base_term, &mut process),
        &mut process
    );
}

#[test]
fn with_tuple_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let tuple_term = Term::slice_to_tuple(&[], &mut process);
    let base_term: Term = 16.into_process(&mut process);

    assert_bad_argument!(
        erlang::binary_to_integer_2(tuple_term, base_term, &mut process),
        &mut process
    );
}

#[test]
fn with_map_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let map_term = Term::slice_to_map(&[], &mut process);
    let base_term: Term = 16.into_process(&mut process);

    assert_bad_argument!(
        erlang::binary_to_integer_2(map_term, base_term, &mut process),
        &mut process
    );
}

#[test]
fn with_heap_binary_with_min_small_integer_returns_small_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary("-800000000000000".as_bytes(), &mut process);
    let base_term: Term = 16.into_process(&mut process);

    let integer_result = erlang::binary_to_integer_2(heap_binary_term, base_term, &mut process);

    assert_eq_in_process!(
        integer_result,
        Ok(<BigInt as Num>::from_str_radix("-576460752303423488", 10)
            .unwrap()
            .into_process(&mut process)),
        process
    );
    assert_eq!(integer_result.unwrap().tag(), Tag::SmallInteger);
}

#[test]
fn with_heap_binary_with_max_small_integer_returns_small_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary("7FFFFFFFFFFFFFF".as_bytes(), &mut process);
    let base_term: Term = 16.into_process(&mut process);

    let integer_result = erlang::binary_to_integer_2(heap_binary_term, base_term, &mut process);

    assert_eq_in_process!(
        integer_result,
        Ok(<BigInt as Num>::from_str_radix("576460752303423487", 10)
            .unwrap()
            .into_process(&mut process)),
        process
    );
    assert_eq!(integer_result.unwrap().tag(), Tag::SmallInteger);
}

#[test]
fn with_heap_binary_with_less_than_min_small_integer_returns_big_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary("-800000000000001".as_bytes(), &mut process);
    let base_term: Term = 16.into_process(&mut process);

    let integer_result = erlang::binary_to_integer_2(heap_binary_term, base_term, &mut process);

    assert_eq_in_process!(
        integer_result,
        Ok(<BigInt as Num>::from_str_radix("-576460752303423489", 10)
            .unwrap()
            .into_process(&mut process)),
        process
    );

    let integer = integer_result.unwrap();

    assert_eq!(integer.tag(), Tag::Boxed);

    let unboxed: &Term = integer.unbox_reference();

    assert_eq!(unboxed.tag(), Tag::BigInteger);
}

#[test]
fn with_heap_binary_with_greater_than_max_small_integer_returns_big_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary("800000000000000".as_bytes(), &mut process);
    let base_term: Term = 16.into_process(&mut process);

    let integer_result = erlang::binary_to_integer_2(heap_binary_term, base_term, &mut process);

    assert_eq_in_process!(
        integer_result,
        Ok(<BigInt as Num>::from_str_radix("576460752303423488", 10)
            .unwrap()
            .into_process(&mut process)),
        process
    );

    let integer = integer_result.unwrap();

    assert_eq!(integer.tag(), Tag::Boxed);

    let unboxed: &Term = integer.unbox_reference();

    assert_eq!(unboxed.tag(), Tag::BigInteger);
}

#[test]
fn with_subbinary_with_min_small_integer_returns_small_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // <<1::1, Integer.to_string(-576460752303423488, 16) :: binary>>
    let heap_binary_term = Term::slice_to_binary(
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
    let subbinary_term = Term::subbinary(heap_binary_term, 0, 1, 16, 0, &mut process);
    let base_term: Term = 16.into_process(&mut process);

    let integer_result = erlang::binary_to_integer_2(subbinary_term, base_term, &mut process);

    assert_eq_in_process!(
        integer_result,
        Ok(<BigInt as Num>::from_str_radix("-576460752303423488", 10)
            .unwrap()
            .into_process(&mut process)),
        process
    );
    assert_eq!(integer_result.unwrap().tag(), Tag::SmallInteger);
}

#[test]
fn with_subbinary_with_max_small_integer_returns_small_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // <<1::1, Integer.to_string(576460752303423487, 16) :: binary>>
    let heap_binary_term = Term::slice_to_binary(
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
    let subbinary_term = Term::subbinary(heap_binary_term, 0, 1, 15, 0, &mut process);
    let base_term: Term = 16.into_process(&mut process);

    let integer_result = erlang::binary_to_integer_2(subbinary_term, base_term, &mut process);

    assert_eq_in_process!(
        integer_result,
        Ok(<BigInt as Num>::from_str_radix("576460752303423487", 10)
            .unwrap()
            .into_process(&mut process)),
        process
    );
    assert_eq!(integer_result.unwrap().tag(), Tag::SmallInteger);
}

#[test]
fn with_subbinary_with_less_than_min_small_integer_returns_big_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // <<1::1, Integer.to_string(-576460752303423489, 16) :: binary>>
    let heap_binary_term = Term::slice_to_binary(
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
    let subbinary_term = Term::subbinary(heap_binary_term, 0, 1, 16, 0, &mut process);
    let base_term: Term = 16.into_process(&mut process);

    let integer_result = erlang::binary_to_integer_2(subbinary_term, base_term, &mut process);

    assert_eq_in_process!(
        integer_result,
        Ok(
            <BigInt as Num>::from_str_radix("-5764_60_752_303_423_489", 10)
                .unwrap()
                .into_process(&mut process)
        ),
        process
    );

    let integer = integer_result.unwrap();

    assert_eq!(integer.tag(), Tag::Boxed);

    let unboxed: &Term = integer.unbox_reference();

    assert_eq!(unboxed.tag(), Tag::BigInteger);
}

#[test]
fn with_subbinary_with_greater_than_max_small_integer_returns_big_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // <<1::1, Integer.to_string(576460752303423488, 16) :: binary>>
    let heap_binary_term = Term::slice_to_binary(
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
    let subbinary_term = Term::subbinary(heap_binary_term, 0, 1, 15, 0, &mut process);
    let base_term: Term = 16.into_process(&mut process);

    let integer_result = erlang::binary_to_integer_2(subbinary_term, base_term, &mut process);

    assert_eq_in_process!(
        integer_result,
        Ok(<BigInt as Num>::from_str_radix("576460752303423488", 10)
            .unwrap()
            .into_process(&mut process)),
        process
    );

    let integer = integer_result.unwrap();

    assert_eq!(integer.tag(), Tag::Boxed);

    let unboxed: &Term = integer.unbox_reference();

    assert_eq!(unboxed.tag(), Tag::BigInteger);
}
