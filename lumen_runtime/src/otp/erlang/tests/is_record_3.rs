use super::*;

use crate::process::IntoProcess;

mod with_tuple;

#[test]
fn with_atom_is_false() {
    is_not_record_with_term(|_| Term::str_to_atom("term", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_is_false() {
    is_not_record_with_term(|process| Term::local_reference(&process));
}

#[test]
fn with_empty_list_is_false() {
    is_not_record_with_term(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_is_false() {
    is_not_record_with_term(|process| list_term(&process));
}

#[test]
fn with_small_integer_is_false() {
    is_not_record_with_term(|process| 0.into_process(&process));
}

#[test]
fn with_big_integer_is_false() {
    is_not_record_with_term(|process| (integer::small::MAX + 1).into_process(&process));
}

#[test]
fn with_float_is_false() {
    is_not_record_with_term(|process| 1.0.into_process(&process));
}

#[test]
fn with_local_pid_is_true() {
    is_not_record_with_term(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_is_true() {
    is_not_record_with_term(|process| Term::external_pid(1, 0, 0, &process).unwrap());
}

#[test]
fn with_map_is_false() {
    is_not_record_with_term(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_is_false() {
    is_not_record_with_term(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_is_false() {
    is_not_record_with_term(|process| bitstring!(1 :: 1, &process));
}

fn is_not_record_with_term<T>(term: T)
where
    T: FnOnce(&Process) -> Term,
{
    with_process(|process| {
        let record_tag = Term::str_to_atom("record_tag", DoNotCare).unwrap();
        let size = 1.into_process(&process);

        assert_eq!(
            erlang::is_record_3(term(&process), record_tag, size),
            Ok(false.into())
        );
    });
}
