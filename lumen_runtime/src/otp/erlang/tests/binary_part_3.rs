use super::*;

use std::sync::{Arc, RwLock};

use num_traits::Num;

use crate::environment::{self, Environment};
use crate::process::IntoProcess;

#[test]
fn with_atom_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::binary_part_3(
            atom_term,
            0.into_process(&mut process),
            0.into_process(&mut process),
            &mut process
        ),
        &mut process
    );
}

#[test]
fn with_empty_list_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    assert_bad_argument!(
        erlang::binary_part_3(
            Term::EMPTY_LIST,
            0.into_process(&mut process),
            0.into_process(&mut process),
            &mut process
        ),
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
        erlang::binary_part_3(
            list_term,
            0.into_process(&mut process),
            0.into_process(&mut process),
            &mut process
        ),
        &mut process
    );
}

#[test]
fn with_small_integer_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let small_integer_term: Term = 0.into_process(&mut process);

    assert_bad_argument!(
        erlang::binary_part_3(
            small_integer_term,
            0.into_process(&mut process),
            0.into_process(&mut process),
            &mut process
        ),
        &mut process
    );
}

#[test]
fn with_big_integer_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let big_integer_term: Term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
        .unwrap()
        .into_process(&mut process);

    assert_bad_argument!(
        erlang::binary_part_3(
            big_integer_term,
            0.into_process(&mut process),
            0.into_process(&mut process),
            &mut process
        ),
        &mut process
    );
}

#[test]
fn with_float_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let float_term: Term = 1.0.into_process(&mut process);

    assert_bad_argument!(
        erlang::binary_part_3(
            float_term,
            0.into_process(&mut process),
            0.into_process(&mut process),
            &mut process
        ),
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
        erlang::binary_part_3(
            local_pid_term,
            0.into_process(&mut process),
            0.into_process(&mut process),
            &mut process
        ),
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
        erlang::binary_part_3(
            external_pid_term,
            0.into_process(&mut process),
            0.into_process(&mut process),
            &mut process
        ),
        &mut process
    );
}

#[test]
fn with_tuple_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let tuple_term = Term::slice_to_tuple(
        &[0.into_process(&mut process), 1.into_process(&mut process)],
        &mut process,
    );

    assert_bad_argument!(
        erlang::binary_part_3(
            tuple_term,
            0.into_process(&mut process),
            0.into_process(&mut process),
            &mut process
        ),
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
        erlang::binary_part_3(
            map_term,
            0.into_process(&mut process),
            0.into_process(&mut process),
            &mut process
        ),
        &mut process
    );
}

#[test]
fn with_heap_binary_without_integer_start_without_integer_length_returns_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary(&[], &mut process);
    let start_term = Term::slice_to_tuple(
        &[0.into_process(&mut process), 0.into_process(&mut process)],
        &mut process,
    );
    let length_term = Term::str_to_atom("all", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::binary_part_3(heap_binary_term, start_term, length_term, &mut process),
        &mut process
    );
}

#[test]
fn with_heap_binary_without_integer_start_with_integer_length_returns_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary(&[], &mut process);
    let start_term = 0.into_process(&mut process);
    let length_term = Term::str_to_atom("all", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::binary_part_3(heap_binary_term, start_term, length_term, &mut process),
        &mut process
    );
}

#[test]
fn with_heap_binary_with_integer_start_without_integer_length_returns_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary(&[], &mut process);
    let start_term = 0.into_process(&mut process);
    let length_term = Term::str_to_atom("all", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::binary_part_3(heap_binary_term, start_term, length_term, &mut process),
        &mut process
    );
}

#[test]
fn with_heap_binary_with_negative_start_with_valid_length_returns_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary(&[], &mut process);
    let start_term = (-1isize).into_process(&mut process);
    let length_term = 0.into_process(&mut process);

    assert_bad_argument!(
        erlang::binary_part_3(heap_binary_term, start_term, length_term, &mut process),
        &mut process
    );
}

#[test]
fn with_heap_binary_with_start_greater_than_size_with_non_negative_length_returns_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary(&[], &mut process);
    let start_term = 1.into_process(&mut process);
    let length_term = 0.into_process(&mut process);

    assert_bad_argument!(
        erlang::binary_part_3(heap_binary_term, start_term, length_term, &mut process),
        &mut process
    );
}

#[test]
fn with_heap_binary_with_start_less_than_size_with_negative_length_past_start_returns_bad_argument()
{
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary(&[0], &mut process);
    let start_term = 0.into_process(&mut process);
    let length_term = (-1isize).into_process(&mut process);

    assert_bad_argument!(
        erlang::binary_part_3(heap_binary_term, start_term, length_term, &mut process),
        &mut process
    );
}

#[test]
fn with_heap_binary_with_start_less_than_size_with_positive_length_past_end_returns_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary(&[0], &mut process);
    let start_term = 0.into_process(&mut process);
    let length_term = 2.into_process(&mut process);

    assert_bad_argument!(
        erlang::binary_part_3(heap_binary_term, start_term, length_term, &mut process),
        &mut process
    );
}

#[test]
fn with_heap_binary_with_zero_start_and_size_length_returns_binary() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary(&[0], &mut process);
    let start_term = 0.into_process(&mut process);
    let length_term = 1.into_process(&mut process);

    assert_eq_in_process!(
        erlang::binary_part_3(heap_binary_term, start_term, length_term, &mut process),
        Ok(heap_binary_term),
        process
    );

    let returned_binary =
        erlang::binary_part_3(heap_binary_term, start_term, length_term, &mut process).unwrap();

    assert_eq!(returned_binary.tagged, heap_binary_term.tagged);
}

#[test]
fn with_heap_binary_with_size_start_and_negative_size_length_returns_binary() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary(&[0], &mut process);
    let start_term = 1.into_process(&mut process);
    let length_term = (-1isize).into_process(&mut process);

    assert_eq_in_process!(
        erlang::binary_part_3(heap_binary_term, start_term, length_term, &mut process),
        Ok(heap_binary_term),
        process
    );

    let returned_binary =
        erlang::binary_part_3(heap_binary_term, start_term, length_term, &mut process).unwrap();

    assert_eq!(returned_binary.tagged, heap_binary_term.tagged);
}

#[test]
fn with_heap_binary_with_positive_start_and_negative_length_returns_subbinary() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary(&[0, 1], &mut process);
    let start_term = 1.into_process(&mut process);
    let length_term = (-1isize).into_process(&mut process);

    assert_eq_in_process!(
        erlang::binary_part_3(heap_binary_term, start_term, length_term, &mut process),
        Ok(Term::slice_to_binary(&[0], &mut process)),
        process
    );

    let returned_boxed =
        erlang::binary_part_3(heap_binary_term, start_term, length_term, &mut process).unwrap();

    assert_eq!(returned_boxed.tag(), Tag::Boxed);

    let returned_unboxed: &Term = returned_boxed.unbox_reference();

    assert_eq!(returned_unboxed.tag(), Tag::Subbinary);
}

#[test]
fn with_heap_binary_with_positive_start_and_positive_length_returns_subbinary() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary(&[0, 1], &mut process);
    let start_term = 1.into_process(&mut process);
    let length_term = 1.into_process(&mut process);

    assert_eq_in_process!(
        erlang::binary_part_3(heap_binary_term, start_term, length_term, &mut process),
        Ok(Term::slice_to_binary(&[1], &mut process)),
        process
    );

    let returned_boxed =
        erlang::binary_part_3(heap_binary_term, start_term, length_term, &mut process).unwrap();

    assert_eq!(returned_boxed.tag(), Tag::Boxed);

    let returned_unboxed: &Term = returned_boxed.unbox_reference();

    assert_eq!(returned_unboxed.tag(), Tag::Subbinary);
}

#[test]
fn with_subbinary_without_integer_start_without_integer_length_returns_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
    let start_term = Term::slice_to_tuple(
        &[0.into_process(&mut process), 0.into_process(&mut process)],
        &mut process,
    );
    let length_term = Term::str_to_atom("all", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::binary_part_3(subbinary_term, start_term, length_term, &mut process),
        &mut process
    );
}

#[test]
fn with_subbinary_without_integer_start_with_integer_length_returns_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
    let start_term = 0.into_process(&mut process);
    let length_term = Term::str_to_atom("all", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::binary_part_3(subbinary_term, start_term, length_term, &mut process),
        &mut process
    );
}

#[test]
fn with_subbinary_with_integer_start_without_integer_length_returns_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
    let start_term = 0.into_process(&mut process);
    let length_term = Term::str_to_atom("all", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::binary_part_3(subbinary_term, start_term, length_term, &mut process),
        &mut process
    );
}

#[test]
fn with_subbinary_with_negative_start_with_valid_length_returns_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
    let start_term = (-1isize).into_process(&mut process);
    let length_term = 0.into_process(&mut process);

    assert_bad_argument!(
        erlang::binary_part_3(subbinary_term, start_term, length_term, &mut process),
        &mut process
    );
}

#[test]
fn with_subbinary_with_start_greater_than_size_with_non_negative_length_returns_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 7, 0, 1, &mut process);
    let start_term = 1.into_process(&mut process);
    let length_term = 0.into_process(&mut process);

    assert_bad_argument!(
        erlang::binary_part_3(subbinary_term, start_term, length_term, &mut process),
        &mut process
    );
}

#[test]
fn with_subbinary_with_start_less_than_size_with_negative_length_past_start_returns_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
    let start_term = 0.into_process(&mut process);
    let length_term = (-1isize).into_process(&mut process);

    assert_bad_argument!(
        erlang::binary_part_3(subbinary_term, start_term, length_term, &mut process),
        &mut process
    );
}

#[test]
fn with_subbinary_with_start_less_than_size_with_positive_length_past_end_returns_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 7, 1, 1, &mut process);
    let start_term = 0.into_process(&mut process);
    let length_term = 2.into_process(&mut process);

    assert_bad_argument!(
        erlang::binary_part_3(subbinary_term, start_term, length_term, &mut process),
        &mut process
    );
}

#[test]
fn with_subbinary_with_zero_start_and_byte_count_length_returns_new_subbinary_with_bytes() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
    let start_term = 0.into_process(&mut process);
    let length_term = 2.into_process(&mut process);

    assert_eq_in_process!(
        erlang::binary_part_3(subbinary_term, start_term, length_term, &mut process),
        Ok(Term::subbinary(binary_term, 0, 7, 2, 0, &mut process)),
        process
    );
}

#[test]
fn with_subbinary_with_byte_count_start_and_negative_byte_count_length_returns_new_subbinary_with_bytes(
) {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
    let start_term = 2.into_process(&mut process);
    let length_term = (-2isize).into_process(&mut process);

    assert_eq_in_process!(
        erlang::binary_part_3(subbinary_term, start_term, length_term, &mut process),
        Ok(Term::subbinary(binary_term, 0, 7, 2, 0, &mut process)),
        process
    );
}

#[test]
fn with_subbinary_with_positive_start_and_negative_length_returns_subbinary() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary_term = Term::slice_to_binary(&[0b0000_00001, 0b1111_1110], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 7, 1, 0, &mut process);
    let start_term = 1.into_process(&mut process);
    let length_term = (-1isize).into_process(&mut process);

    assert_eq_in_process!(
        erlang::binary_part_3(subbinary_term, start_term, length_term, &mut process),
        Ok(Term::slice_to_binary(&[0b1111_1111], &mut process)),
        process
    );

    let returned_boxed =
        erlang::binary_part_3(subbinary_term, start_term, length_term, &mut process).unwrap();

    assert_eq!(returned_boxed.tag(), Tag::Boxed);

    let returned_unboxed: &Term = returned_boxed.unbox_reference();

    assert_eq!(returned_unboxed.tag(), Tag::Subbinary);
}

#[test]
fn with_subbinary_with_positive_start_and_positive_length_returns_subbinary() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let subbinary_term: Term = process.subbinary(binary_term, 0, 7, 2, 1).into();
    let start_term = 1.into_process(&mut process);
    let length_term = 1.into_process(&mut process);

    assert_eq_in_process!(
        erlang::binary_part_3(subbinary_term, start_term, length_term, &mut process),
        Ok(Term::slice_to_binary(&[0b0101_0101], &mut process)),
        process
    );

    let returned_boxed =
        erlang::binary_part_3(subbinary_term, start_term, length_term, &mut process).unwrap();

    assert_eq!(returned_boxed.tag(), Tag::Boxed);

    let returned_unboxed: &Term = returned_boxed.unbox_reference();

    assert_eq!(returned_unboxed.tag(), Tag::Subbinary);
}
