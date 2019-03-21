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
    let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::binary_to_term_1(atom_term, &mut process),
        &mut process
    );
}

#[test]
fn with_empty_list_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    assert_bad_argument!(
        erlang::binary_to_term_1(Term::EMPTY_LIST, &mut process),
        &mut process
    );
}

#[test]
fn with_list_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list_term = list_term(&mut process);

    assert_bad_argument!(
        erlang::binary_to_term_1(list_term, &mut process),
        &mut process
    );
}

#[test]
fn with_small_integer_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let small_integer_term = 0usize.into_process(&mut process);

    assert_bad_argument!(
        erlang::binary_to_term_1(small_integer_term, &mut process),
        &mut process
    );
}

#[test]
fn with_big_integer_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
        .unwrap()
        .into_process(&mut process);

    assert_bad_argument!(
        erlang::binary_to_term_1(big_integer_term, &mut process),
        &mut process
    );
}

#[test]
fn with_float_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let float_term = 1.0.into_process(&mut process);

    assert_bad_argument!(
        erlang::binary_to_term_1(float_term, &mut process),
        &mut process
    );
}

#[test]
fn with_local_pid_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let local_pid_term = Term::local_pid(0, 0, &mut process).unwrap();

    assert_bad_argument!(
        erlang::binary_to_term_1(local_pid_term, &mut process),
        &mut process
    );
}

#[test]
fn with_external_pid_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

    assert_bad_argument!(
        erlang::binary_to_term_1(external_pid_term, &mut process),
        &mut process
    );
}

#[test]
fn with_tuple_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let tuple_term = Term::slice_to_tuple(&[], &mut process);

    assert_bad_argument!(
        erlang::binary_to_term_1(tuple_term, &mut process),
        &mut process
    );
}

#[test]
fn with_map_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let map_term = Term::slice_to_map(&[], &mut process);

    assert_bad_argument!(
        erlang::binary_to_term_1(map_term, &mut process),
        &mut process
    );
}

#[test]
fn with_heap_binary_encoding_atom_returns_atom() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // :erlang.term_to_binary(:atom)
    let heap_binary_term =
        Term::slice_to_binary(&[131, 100, 0, 4, 97, 116, 111, 109], &mut process);

    assert_eq_in_process!(
        erlang::binary_to_term_1(heap_binary_term, &mut process),
        Term::str_to_atom("atom", Existence::DoNotCare, &mut process),
        process
    );
}

#[test]
fn with_heap_binary_encoding_empty_list_returns_empty_list() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // :erlang.term_to_binary([])
    let heap_binary_term = Term::slice_to_binary(&[131, 106], &mut process);

    assert_eq_in_process!(
        erlang::binary_to_term_1(heap_binary_term, &mut process),
        Ok(Term::EMPTY_LIST),
        process
    );
}

#[test]
fn with_heap_binary_encoding_list_returns_list() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // :erlang.term_to_binary([:zero, 1])
    let heap_binary_term = Term::slice_to_binary(
        &[
            131, 108, 0, 0, 0, 2, 100, 0, 4, 122, 101, 114, 111, 97, 1, 106,
        ],
        &mut process,
    );

    assert_eq_in_process!(
        erlang::binary_to_term_1(heap_binary_term, &mut process),
        Ok(Term::cons(
            Term::str_to_atom("zero", Existence::DoNotCare, &mut process).unwrap(),
            Term::cons(1.into_process(&mut process), Term::EMPTY_LIST, &mut process),
            &mut process
        )),
        process
    );
}

#[test]
fn with_heap_binary_encoding_small_integer_returns_small_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // :erlang.term_to_binary(0)
    let heap_binary_term = Term::slice_to_binary(&[131, 97, 0], &mut process);

    assert_eq_in_process!(
        erlang::binary_to_term_1(heap_binary_term, &mut process),
        Ok(0.into_process(&mut process)),
        process
    );
}

#[test]
fn with_heap_binary_encoding_integer_returns_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // :erlang.term_to_binary(-2147483648)
    let heap_binary_term = Term::slice_to_binary(&[131, 98, 128, 0, 0, 0], &mut process);

    assert_eq_in_process!(
        erlang::binary_to_term_1(heap_binary_term, &mut process),
        Ok((-2147483648_isize).into_process(&mut process)),
        process
    );
}

#[test]
fn with_heap_binary_encoding_new_float_returns_float() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // :erlang.term_to_binary(1.0)
    let heap_binary_term =
        Term::slice_to_binary(&[131, 70, 63, 240, 0, 0, 0, 0, 0, 0], &mut process);

    assert_eq_in_process!(
        erlang::binary_to_term_1(heap_binary_term, &mut process),
        Ok(1.0.into_process(&mut process)),
        process
    );
}

#[test]
fn with_heap_binary_encoding_small_tuple_returns_tuple() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // :erlang.term_to_binary({:zero, 1})
    let heap_binary_term = Term::slice_to_binary(
        &[131, 104, 2, 100, 0, 4, 122, 101, 114, 111, 97, 1],
        &mut process,
    );

    assert_eq_in_process!(
        erlang::binary_to_term_1(heap_binary_term, &mut process),
        Ok(Term::slice_to_tuple(
            &[
                Term::str_to_atom("zero", Existence::DoNotCare, &mut process).unwrap(),
                1.into_process(&mut process)
            ],
            &mut process
        )),
        process
    );
}

#[test]
fn with_heap_binary_encoding_byte_list_returns_list() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // :erlang.term_to_binary([?0, ?1])
    let heap_binary_term = Term::slice_to_binary(&[131, 107, 0, 2, 48, 49], &mut process);

    assert_eq_in_process!(
        erlang::binary_to_term_1(heap_binary_term, &mut process),
        Ok(Term::cons(
            48.into_process(&mut process),
            Term::cons(
                49.into_process(&mut process),
                Term::EMPTY_LIST,
                &mut process
            ),
            &mut process
        )),
        process
    );
}

#[test]
fn with_heap_binary_encoding_binary_returns_binary() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // :erlang.term_to_binary(<<0, 1>>)
    let heap_binary_term = Term::slice_to_binary(&[131, 109, 0, 0, 0, 2, 0, 1], &mut process);

    assert_eq_in_process!(
        erlang::binary_to_term_1(heap_binary_term, &mut process),
        Ok(Term::slice_to_binary(&[0, 1], &mut process)),
        process
    );
}

#[test]
fn with_heap_binary_encoding_small_big_integer_returns_big_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // :erlang.term_to_binary(4294967295)
    let heap_binary_term =
        Term::slice_to_binary(&[131, 110, 4, 0, 255, 255, 255, 255], &mut process);

    assert_eq_in_process!(
        erlang::binary_to_term_1(heap_binary_term, &mut process),
        Ok(4294967295_usize.into_process(&mut process)),
        process
    );
}

#[test]
fn with_heap_binary_encoding_bit_string_returns_subbinary() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // :erlang.term_to_binary(<<1, 2::3>>)
    let heap_binary_term = Term::slice_to_binary(&[131, 77, 0, 0, 0, 2, 3, 1, 64], &mut process);

    assert_eq_in_process!(
        erlang::binary_to_term_1(heap_binary_term, &mut process),
        Ok(Term::subbinary(
            Term::slice_to_binary(&[1, 0b010_00000], &mut process),
            0,
            0,
            1,
            3,
            &mut process
        )),
        process,
    );
}

#[test]
fn with_heap_binary_encoding_small_atom_utf8_returns_atom() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // :erlang.term_to_binary(:"😈")
    let heap_binary_term = Term::slice_to_binary(&[131, 119, 4, 240, 159, 152, 136], &mut process);

    assert_eq_in_process!(
        erlang::binary_to_term_1(heap_binary_term, &mut process),
        Ok(Term::str_to_atom("😈", Existence::DoNotCare, &mut process).unwrap()),
        process,
    );
}

#[test]
fn with_subbinary_encoding_atom_returns_atom() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // <<1::1, :erlang.term_to_binary(:atom) :: binary>>
    let original_term = Term::slice_to_binary(
        &[193, 178, 0, 2, 48, 186, 55, 182, 0b1000_0000],
        &mut process,
    );
    let subbinary_term = Term::subbinary(original_term, 0, 1, 8, 0, &mut process);

    assert_eq_in_process!(
        erlang::binary_to_term_1(subbinary_term, &mut process),
        Term::str_to_atom("atom", Existence::DoNotCare, &mut process),
        process
    );
}

#[test]
fn with_subbinary_encoding_empty_list_returns_empty_list() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // <<1::1, :erlang.term_to_binary([]) :: binary>>
    let original_term = Term::slice_to_binary(&[193, 181, 0b0000_0000], &mut process);
    let subbinary_term = Term::subbinary(original_term, 0, 1, 2, 0, &mut process);

    assert_eq_in_process!(
        erlang::binary_to_term_1(subbinary_term, &mut process),
        Ok(Term::EMPTY_LIST),
        process
    );
}

#[test]
fn with_subbinary_encoding_list_returns_list() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // <<1::1, :erlang.term_to_binary([:zero, 1]) :: binary>>
    let original_term = Term::slice_to_binary(
        &[
            193, 182, 0, 0, 0, 1, 50, 0, 2, 61, 50, 185, 55, 176, 128, 181, 0,
        ],
        &mut process,
    );
    let subbinary_term = Term::subbinary(original_term, 0, 1, 16, 0, &mut process);

    assert_eq_in_process!(
        erlang::binary_to_term_1(subbinary_term, &mut process),
        Ok(Term::cons(
            Term::str_to_atom("zero", Existence::DoNotCare, &mut process).unwrap(),
            Term::cons(1.into_process(&mut process), Term::EMPTY_LIST, &mut process),
            &mut process
        )),
        process
    );
}

#[test]
fn with_subbinary_encoding_small_integer_returns_small_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // <<1::1, :erlang.term_to_binary(0) :: binary>>
    let original_term = Term::slice_to_binary(&[193, 176, 128, 0], &mut process);
    let subbinary_term = Term::subbinary(original_term, 0, 1, 3, 0, &mut process);

    assert_eq_in_process!(
        erlang::binary_to_term_1(subbinary_term, &mut process),
        Ok(0.into_process(&mut process)),
        process
    );
}

#[test]
fn with_subbinary_encoding_integer_returns_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // <<1::1, :erlang.term_to_binary(-2147483648) :: binary>>
    let original_term = Term::slice_to_binary(&[193, 177, 64, 0, 0, 0, 0], &mut process);
    let subbinary_term = Term::subbinary(original_term, 0, 1, 6, 0, &mut process);

    assert_eq_in_process!(
        erlang::binary_to_term_1(subbinary_term, &mut process),
        Ok((-2147483648_isize).into_process(&mut process)),
        process
    );
}

#[test]
fn with_subbinary_encoding_new_float_returns_float() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // <<1::1, :erlang.term_to_binary(1.0) :: binary>>
    let original_term =
        Term::slice_to_binary(&[193, 163, 31, 248, 0, 0, 0, 0, 0, 0, 0], &mut process);
    let subbinary_term = Term::subbinary(original_term, 0, 1, 10, 0, &mut process);

    assert_eq_in_process!(
        erlang::binary_to_term_1(subbinary_term, &mut process),
        Ok(1.0.into_process(&mut process)),
        process
    );
}

#[test]
fn with_subbinary_encoding_small_tuple_returns_tuple() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // <<1::1, :erlang.term_to_binary({:zero, 1}) :: binary>>
    let original_term = Term::slice_to_binary(
        &[
            193,
            180,
            1,
            50,
            0,
            2,
            61,
            50,
            185,
            55,
            176,
            128,
            0b1000_0000,
        ],
        &mut process,
    );
    let subbinary_term = Term::subbinary(original_term, 0, 1, 12, 0, &mut process);

    assert_eq_in_process!(
        erlang::binary_to_term_1(subbinary_term, &mut process),
        Ok(Term::slice_to_tuple(
            &[
                Term::str_to_atom("zero", Existence::DoNotCare, &mut process).unwrap(),
                1.into_process(&mut process)
            ],
            &mut process
        )),
        process
    );
}

#[test]
fn with_subbinary_encoding_byte_list_returns_list() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // <<1::1, :erlang.term_to_binary([?0, ?1]) :: binary>>
    let original_term =
        Term::slice_to_binary(&[193, 181, 128, 1, 24, 24, 0b1000_0000], &mut process);
    let subbinary_term = Term::subbinary(original_term, 0, 1, 6, 0, &mut process);

    assert_eq_in_process!(
        erlang::binary_to_term_1(subbinary_term, &mut process),
        Ok(Term::cons(
            48.into_process(&mut process),
            Term::cons(
                49.into_process(&mut process),
                Term::EMPTY_LIST,
                &mut process
            ),
            &mut process
        )),
        process
    );
}

#[test]
fn with_subbinary_encoding_binary_returns_binary() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // <<1::1, :erlang.term_to_binary(<<0, 1>>) :: binary>>
    let original_term =
        Term::slice_to_binary(&[193, 182, 128, 0, 0, 1, 0, 0, 0b1000_0000], &mut process);
    let subbinary_term = Term::subbinary(original_term, 0, 1, 8, 0, &mut process);

    assert_eq_in_process!(
        erlang::binary_to_term_1(subbinary_term, &mut process),
        Ok(Term::slice_to_binary(&[0, 1], &mut process)),
        process
    );
}

#[test]
fn with_subbinary_encoding_small_big_integer_returns_big_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // <<1::1, :erlang.term_to_binary(4294967295) :: binary>>
    let original_term = Term::slice_to_binary(
        &[193, 183, 2, 0, 127, 255, 255, 255, 0b1000_0000],
        &mut process,
    );
    let subbinary_term = Term::subbinary(original_term, 0, 1, 8, 0, &mut process);

    assert_eq_in_process!(
        erlang::binary_to_term_1(subbinary_term, &mut process),
        Ok(4294967295_usize.into_process(&mut process)),
        process
    );
}

#[test]
fn with_subbinary_encoding_bit_string_returns_subbinary() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // <<1::1, :erlang.term_to_binary(<<1, 2::3>>) :: binary>>
    let original_term =
        Term::slice_to_binary(&[193, 166, 128, 0, 0, 1, 1, 128, 160, 0], &mut process);
    let subbinary_term = Term::subbinary(original_term, 0, 1, 9, 0, &mut process);

    assert_eq_in_process!(
        erlang::binary_to_term_1(subbinary_term, &mut process),
        Ok(Term::subbinary(
            Term::slice_to_binary(&[1, 0b010_00000], &mut process),
            0,
            0,
            1,
            3,
            &mut process
        )),
        process,
    );
}

#[test]
fn with_subbinary_encoding_small_atom_utf8_returns_atom() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // <<1::1, :erlang.term_to_binary(:"😈") :: binary>>
    let original_term = Term::slice_to_binary(&[193, 187, 130, 120, 79, 204, 68, 0], &mut process);
    let subbinary_term = Term::subbinary(original_term, 0, 1, 7, 0, &mut process);

    assert_eq_in_process!(
        erlang::binary_to_term_1(subbinary_term, &mut process),
        Ok(Term::str_to_atom("😈", Existence::DoNotCare, &mut process).unwrap()),
        process,
    );
}
