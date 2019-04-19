use super::*;

use num_traits::Num;

use crate::process::IntoProcess;

mod with_atom_record_tag;

#[test]
fn with_local_reference_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|process| Term::local_reference(&process));
}

#[test]
fn with_empty_list_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|process| list_term(&process));
}

#[test]
fn with_small_integer_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|process| 0.into_process(&process));
}

#[test]
fn with_big_integer_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|process| {
        <BigInt as Num>::from_str_radix("576460752303423489", 10)
            .unwrap()
            .into_process(&process)
    });
}

#[test]
fn with_float_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|process| 1.0.into_process(&process));
}

#[test]
fn with_local_pid_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|process| Term::external_pid(1, 0, 0, &process).unwrap());
}

#[test]
fn with_tuple_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|process| {
        let original = Term::slice_to_binary(&[129, 0b0000_0000], &process);
        Term::subbinary(original, 0, 1, 1, 0, &process)
    });
}

fn with_record_tag_errors_badarg<F>(record_tag: F)
where
    F: FnOnce(&Process) -> Term,
{
    super::errors_badarg(|process| {
        let record_tag = record_tag(&process);
        let term = Term::slice_to_tuple(&[record_tag], &process);
        let size = 1.into_process(&process);

        erlang::is_record_3(term, record_tag, size)
    });
}
