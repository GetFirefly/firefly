use super::*;

use num_traits::Num;

use std::sync::{Arc, RwLock};

use crate::environment::{self, Environment};

#[test]
fn with_atom_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list = Term::str_to_atom("list", DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::list_to_existing_atom_1(list, &mut process),
        &mut process
    );
}

#[test]
fn with_local_reference_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list = Term::local_reference(&mut process);

    assert_bad_argument!(
        erlang::list_to_existing_atom_1(list, &mut process),
        &mut process
    );
}

#[test]
fn with_empty_list() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list = Term::EMPTY_LIST;

    assert_bad_argument!(
        erlang::list_to_existing_atom_1(list, &mut process),
        &mut process
    );

    let existing_atom = Term::str_to_atom("", DoNotCare, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::list_to_existing_atom_1(list, &mut process),
        Ok(existing_atom),
        &mut process
    );
}

#[test]
fn with_improper_list_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list = Term::cons(
        'a'.into_process(&mut process),
        'b'.into_process(&mut process),
        &mut process,
    );

    assert_bad_argument!(
        erlang::list_to_existing_atom_1(list, &mut process),
        &mut process
    );
}

#[test]
fn with_list_encoding_utf8() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    let str1 = "atom";
    let char_list1 = Term::str_to_char_list(str1, &mut process);

    assert_bad_argument!(
        erlang::list_to_existing_atom_1(char_list1, &mut process),
        &mut process
    );

    let existing_atom1 = Term::str_to_atom(str1, DoNotCare, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::list_to_existing_atom_1(char_list1, &mut process),
        Ok(existing_atom1),
        process,
    );

    let str2 = "José";
    let char_list2 = Term::str_to_char_list(str2, &mut process);

    assert_bad_argument!(
        erlang::list_to_existing_atom_1(char_list2, &mut process),
        &mut process
    );

    let existing_atom2 = Term::str_to_atom(str2, DoNotCare, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::list_to_existing_atom_1(char_list2, &mut process),
        Ok(existing_atom2),
        process,
    );

    let str3 = "😈";
    let char_list3 = Term::str_to_char_list(str3, &mut process);

    assert_bad_argument!(
        erlang::list_to_existing_atom_1(char_list3, &mut process),
        &mut process
    );

    let existing_atom3 = Term::str_to_atom(str3, DoNotCare, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::list_to_existing_atom_1(char_list3, &mut process),
        Ok(existing_atom3),
        process,
    );
}

#[test]
fn with_list_not_encoding_ut8() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list = Term::cons(
        // from https://doc.rust-lang.org/std/char/fn.from_u32.html
        0x110000.into_process(&mut process),
        Term::EMPTY_LIST,
        &mut process,
    );

    assert_bad_argument!(
        erlang::list_to_existing_atom_1(list, &mut process),
        &mut process
    );
}

#[test]
fn with_small_integer_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list = 0.into_process(&mut process);

    assert_bad_argument!(
        erlang::list_to_existing_atom_1(list, &mut process),
        &mut process
    );
}

#[test]
fn with_big_integer_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list = <BigInt as Num>::from_str_radix("576460752303423489", 10)
        .unwrap()
        .into_process(&mut process);

    assert_bad_argument!(
        erlang::list_to_existing_atom_1(list, &mut process),
        &mut process
    );
}

#[test]
fn with_float_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list = 1.0.into_process(&mut process);

    assert_bad_argument!(
        erlang::list_to_existing_atom_1(list, &mut process),
        &mut process
    );
}

#[test]
fn with_local_pid_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list = Term::local_pid(0, 0, &mut process).unwrap();

    assert_bad_argument!(
        erlang::list_to_existing_atom_1(list, &mut process),
        &mut process
    );
}

#[test]
fn with_external_pid_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list = Term::external_pid(1, 0, 0, &mut process).unwrap();

    assert_bad_argument!(
        erlang::list_to_existing_atom_1(list, &mut process),
        &mut process
    );
}

#[test]
fn with_tuple_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list = Term::slice_to_tuple(&[], &mut process);

    assert_bad_argument!(
        erlang::list_to_existing_atom_1(list, &mut process),
        &mut process
    );
}

#[test]
fn with_map_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let map_term = Term::slice_to_map(&[], &mut process);

    assert_bad_argument!(
        erlang::list_to_existing_atom_1(map_term, &mut process),
        &mut process
    );
}

#[test]
fn with_heap_binary_is_false() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary(&[], &mut process);

    assert_bad_argument!(
        erlang::list_to_existing_atom_1(heap_binary_term, &mut process),
        &mut process
    );
}

#[test]
fn with_subbinary_is_false() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);

    assert_bad_argument!(
        erlang::list_to_existing_atom_1(subbinary_term, &mut process),
        &mut process
    );
}
