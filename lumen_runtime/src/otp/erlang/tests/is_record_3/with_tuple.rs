use super::*;

use num_traits::Num;

use crate::process::IntoProcess;

mod with_atom_record_tag;

#[test]
fn with_local_reference_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|mut process| Term::local_reference(&mut process));
}

#[test]
fn with_empty_list_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|mut process| list_term(&mut process));
}

#[test]
fn with_small_integer_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|mut process| 0.into_process(&mut process));
}

#[test]
fn with_big_integer_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|mut process| {
        <BigInt as Num>::from_str_radix("576460752303423489", 10)
            .unwrap()
            .into_process(&mut process)
    });
}

#[test]
fn with_float_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|mut process| 1.0.into_process(&mut process));
}

#[test]
fn with_local_pid_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|mut process| Term::external_pid(1, 0, 0, &mut process).unwrap());
}

#[test]
fn with_tuple_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|mut process| Term::slice_to_tuple(&[], &mut process));
}

#[test]
fn with_map_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|mut process| Term::slice_to_map(&[], &mut process));
}

#[test]
fn with_heap_binary_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|mut process| Term::slice_to_binary(&[], &mut process));
}

#[test]
fn with_subbinary_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|mut process| {
        let original = Term::slice_to_binary(&[129, 0b0000_0000], &mut process);
        Term::subbinary(original, 0, 1, 1, 0, &mut process)
    });
}

fn with_record_tag_errors_badarg<F>(record_tag: F)
where
    F: FnOnce(&mut Process) -> Term,
{
    super::errors_badarg(|mut process| {
        let record_tag = record_tag(&mut process);
        let term = Term::slice_to_tuple(&[record_tag], &mut process);
        let size = 1.into_process(&mut process);

        erlang::is_record_3(term, record_tag, size)
    });
}
