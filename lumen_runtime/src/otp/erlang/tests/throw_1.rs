use super::*;

use num_traits::Num;

use crate::process::IntoProcess;

#[test]
fn with_atom_throws_with_atom_reason() {
    throws(|_| Term::str_to_atom("reason", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_throws_with_local_reference() {
    throws(|process| Term::local_reference(&process));
}

#[test]
fn with_empty_list_throws_with_empty_list_reason() {
    throws(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_throws_with_list_reason() {
    throws(|process| list_term(&process));
}

#[test]
fn with_small_integer_throws_with_small_integer_reason() {
    throws(|process| 0usize.into_process(&process));
}

#[test]
fn with_big_integer_throws_with_big_integer_reason() {
    throws(|process| {
        <BigInt as Num>::from_str_radix("576460752303423489", 10)
            .unwrap()
            .into_process(&process)
    });
}

#[test]
fn with_float_throws_with_float_reason() {
    throws(|process| 1.0.into_process(&process));
}

#[test]
fn with_local_pid_throws_with_local_pid_reason() {
    throws(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_throws_with_external_pid_reason() {
    throws(|process| Term::external_pid(1, 0, 0, &process).unwrap());
}

#[test]
fn with_tuple_throws_with_tuple_reason() {
    throws(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_throws_with_map_reason() {
    throws(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_throws_with_heap_binary_reason() {
    throws(|process| {
        // :erlang.term_to_binary(:atom)
        Term::slice_to_binary(&[131, 100, 0, 4, 97, 116, 111, 109], &process)
    });
}

#[test]
fn with_subbinary_throws_with_subbinary_reason() {
    throws(|process| {
        // <<1::1, :erlang.term_to_binary(:atom) :: binary>>
        let original =
            Term::slice_to_binary(&[193, 178, 0, 2, 48, 186, 55, 182, 0b1000_0000], &process);
        Term::subbinary(original, 0, 1, 8, 0, &process)
    });
}

fn throws<R>(reason: R)
where
    R: FnOnce(&Process) -> Term,
{
    with_process(|process| {
        let reason = reason(&process);

        assert_throw!(erlang::throw_1(reason), reason);
    })
}
