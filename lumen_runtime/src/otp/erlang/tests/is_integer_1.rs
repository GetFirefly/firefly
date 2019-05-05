use super::*;

use crate::process::IntoProcess;

#[test]
fn with_atom_is_false() {
    let term = Term::str_to_atom("atom", DoNotCare).unwrap();

    assert_eq!(erlang::is_integer_1(term), false.into());
}

#[test]
fn with_local_reference_is_false() {
    with_process(|process| {
        let term = Term::next_local_reference(process);

        assert_eq!(erlang::is_integer_1(term), false.into());
    });
}

#[test]
fn with_empty_list_is_false() {
    let term = Term::EMPTY_LIST;

    assert_eq!(erlang::is_integer_1(term), false.into());
}

#[test]
fn with_list_is_false() {
    with_process(|process| {
        let term = list_term(&process);

        assert_eq!(erlang::is_integer_1(term), false.into());
    });
}

#[test]
fn with_small_integer_is_true() {
    with_process(|process| {
        let term = 0usize.into_process(&process);

        assert_eq!(erlang::is_integer_1(term), true.into());
    });
}

#[test]
fn with_big_integer_is_true() {
    with_process(|process| {
        let term = (integer::small::MAX + 1).into_process(&process);

        assert_eq!(erlang::is_integer_1(term), true.into());
    });
}

#[test]
fn with_float_is_false() {
    with_process(|process| {
        let term = 1.0.into_process(&process);

        assert_eq!(erlang::is_integer_1(term), false.into());
    });
}

#[test]
fn with_local_pid_is_false() {
    let term = Term::local_pid(0, 0).unwrap();

    assert_eq!(erlang::is_integer_1(term), false.into());
}

#[test]
fn with_external_pid_is_false() {
    with_process(|process| {
        let term = Term::external_pid(1, 0, 0, &process).unwrap();

        assert_eq!(erlang::is_integer_1(term), false.into());
    });
}

#[test]
fn with_tuple_is_false() {
    with_process(|process| {
        let term = Term::slice_to_tuple(&[], &process);

        assert_eq!(erlang::is_integer_1(term), false.into());
    });
}

#[test]
fn with_map_is_false() {
    with_process(|process| {
        let term = Term::slice_to_map(&[], &process);

        assert_eq!(erlang::is_integer_1(term), false.into());
    });
}

#[test]
fn with_heap_binary_is_false() {
    with_process(|process| {
        let term = Term::slice_to_binary(&[], &process);

        assert_eq!(erlang::is_integer_1(term), false.into());
    });
}

#[test]
fn with_subbinary_is_false() {
    with_process(|process| {
        let term = bitstring!(1 :: 1, &process);

        assert_eq!(erlang::is_integer_1(term), false.into());
    });
}
