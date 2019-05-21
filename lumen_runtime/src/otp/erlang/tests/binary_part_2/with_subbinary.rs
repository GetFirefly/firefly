use super::*;

mod with_tuple_with_arity_2;

#[test]
fn with_atom_start_length_errors_badarg() {
    with_start_length_errors_badarg(|_| Term::str_to_atom("atom", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_start_length_errors_badarg() {
    with_start_length_errors_badarg(|process| Term::next_local_reference(process));
}

#[test]
fn with_heap_binary_start_length_errors_badarg() {
    with_start_length_errors_badarg(|process| Term::slice_to_binary(&[0], &process));
}

#[test]
fn with_subbinary_start_length_errors_badarg() {
    with_start_length_errors_badarg(|process| {
        let original =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &process);
        Term::subbinary(original, 0, 7, 2, 1, &process)
    });
}

#[test]
fn with_empty_list_start_length_errors_badarg() {
    with_start_length_errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_start_length_errors_badarg() {
    with_start_length_errors_badarg(|process| list_term(&process));
}

#[test]
fn with_small_integer_start_length_errors_badarg() {
    with_start_length_errors_badarg(|process| 0.into_process(&process));
}

#[test]
fn with_big_integer_start_length_errors_badarg() {
    with_start_length_errors_badarg(|process| {
        <BigInt as Num>::from_str_radix("576460752303423489", 10)
            .unwrap()
            .into_process(&process)
    });
}

#[test]
fn with_float_start_length_errors_badarg() {
    with_start_length_errors_badarg(|process| 1.0.into_process(&process));
}

#[test]
fn with_local_pid_start_length_errors_badarg() {
    with_start_length_errors_badarg(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_start_length_errors_badarg() {
    with_start_length_errors_badarg(|process| {
        Term::external_pid(1, 0, 0, &process).unwrap()
    });
}

#[test]
fn with_tuple_without_arity_2_errors_badarg() {
    with_start_length_errors_badarg(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_errors_badarg() {
    with_start_length_errors_badarg(|process| Term::slice_to_map(&[], &process));
}

fn with_start_length_errors_badarg<S>(start_length: S)
    where
        S: FnOnce(&Process) -> Term,
{
    errors_badarg(|process| {
        let original = Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &process);
        let binary: Term = process.subbinary(original, 0, 7, 2, 1).into();
        let start_length = start_length(&process);

        erlang::binary_part_2(binary, start_length, &process)
    })
}
