use super::*;

use num_traits::Num;

use std::sync::{Arc, RwLock};

use crate::environment::{self, Environment};

#[test]
fn with_atom_errors_badarg() {
    errors_badarg(|_| Term::str_to_atom("list", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_errors_badarg() {
    errors_badarg(|mut process| Term::local_reference(&mut process));
}

#[test]
fn with_empty_list() {
    let list = Term::EMPTY_LIST;

    // as `""` can only be entered into the global atom table, can't test with non-existing atom
    let existing_atom = Term::str_to_atom("", DoNotCare).unwrap();

    assert_eq!(erlang::list_to_existing_atom_1(list), Ok(existing_atom));
}

#[test]
fn with_improper_list_errors_badarg() {
    errors_badarg(|mut process| {
        Term::cons(
            'a'.into_process(&mut process),
            'b'.into_process(&mut process),
            &mut process,
        )
    });
}

#[test]
fn with_list_encoding_utf8() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    let string1 = format!("{}:{}:atom", file!(), line!());
    let char_list1 = Term::str_to_char_list(&string1, &mut process);

    assert_badarg!(erlang::list_to_existing_atom_1(char_list1));

    let existing_atom1 = Term::str_to_atom(&string1, DoNotCare).unwrap();

    assert_eq!(
        erlang::list_to_existing_atom_1(char_list1),
        Ok(existing_atom1)
    );

    let string2 = format!("{}:{}:José", file!(), line!());
    let char_list2 = Term::str_to_char_list(&string2, &mut process);

    assert_badarg!(erlang::list_to_existing_atom_1(char_list2));

    let existing_atom2 = Term::str_to_atom(&string2, DoNotCare).unwrap();

    assert_eq!(
        erlang::list_to_existing_atom_1(char_list2),
        Ok(existing_atom2)
    );

    let string3 = format!("{}:{}:😈", file!(), line!());
    let char_list3 = Term::str_to_char_list(&string3, &mut process);

    assert_badarg!(erlang::list_to_existing_atom_1(char_list3));

    let existing_atom3 = Term::str_to_atom(&string3, DoNotCare).unwrap();

    assert_eq!(
        erlang::list_to_existing_atom_1(char_list3),
        Ok(existing_atom3)
    );
}

#[test]
fn with_list_not_encoding_ut8() {
    errors_badarg(|mut process| {
        Term::cons(
            // from https://doc.rust-lang.org/std/char/fn.from_u32.html
            0x110000.into_process(&mut process),
            Term::EMPTY_LIST,
            &mut process,
        )
    });
}

#[test]
fn with_small_integer_errors_badarg() {
    errors_badarg(|mut process| 0.into_process(&mut process));
}

#[test]
fn with_big_integer_errors_badarg() {
    errors_badarg(|mut process| {
        <BigInt as Num>::from_str_radix("576460752303423489", 10)
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
fn with_heap_binary_errors_badmap() {
    errors_badarg(|mut process| Term::slice_to_binary(&[], &mut process));
}

#[test]
fn with_subbinary_errors_badmap() {
    errors_badarg(|mut process| {
        let original =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
        Term::subbinary(original, 0, 7, 2, 1, &mut process)
    });
}

fn errors_badarg<F>(string: F)
where
    F: FnOnce(&mut Process) -> Term,
{
    super::errors_badarg(|mut process| erlang::list_to_existing_atom_1(string(&mut process)));
}
