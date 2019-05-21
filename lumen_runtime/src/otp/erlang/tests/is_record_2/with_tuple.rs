use super::*;

use crate::process::IntoProcess;

#[test]
fn with_atom_record_tag() {
    with_process(|process| {
        let record_tag = Term::str_to_atom("record_tag", DoNotCare).unwrap();
        let term = Term::slice_to_tuple(&[record_tag], &process);

        assert_eq!(erlang::is_record_2(term, record_tag), Ok(true.into()));

        let other_atom = Term::str_to_atom("other_atom", DoNotCare).unwrap();

        assert_eq!(erlang::is_record_2(term, other_atom), Ok(false.into()));
    });
}

#[test]
fn with_local_reference_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|process| Term::next_local_reference(process));
}

#[test]
fn with_float_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|process| 1.0.into_process(&process));
}

#[test]
fn with_local_pid_record_errors_badarg() {
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

        erlang::is_record_2(term, record_tag)
    });
}
