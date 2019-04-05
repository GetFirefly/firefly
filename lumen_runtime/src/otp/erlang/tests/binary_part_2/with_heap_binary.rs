use super::*;

mod with_tuple_with_arity_2;

#[test]
fn with_atom_start_length_errors_badarg() {
    with_start_length_errors_badarg(|_| Term::str_to_atom("atom", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_start_length_errors_badarg() {
    with_start_length_errors_badarg(|mut process| Term::local_reference(&mut process));
}

#[test]
fn with_heap_binary_start_length_errors_badarg() {
    with_start_length_errors_badarg(|mut process| Term::slice_to_binary(&[0], &mut process));
}

#[test]
fn with_subbinary_start_length_errors_badarg() {
    with_start_length_errors_badarg(|mut process| {
        let original =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
        Term::subbinary(original, 0, 7, 2, 1, &mut process)
    });
}

#[test]
fn with_empty_list_start_length_errors_badarg() {
    with_start_length_errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_start_length_errors_badarg() {
    with_start_length_errors_badarg(|mut process| list_term(&mut process));
}

#[test]
fn with_small_integer_start_length_errors_badarg() {
    with_start_length_errors_badarg(|mut process| 0.into_process(&mut process));
}

#[test]
fn with_big_integer_start_length_errors_badarg() {
    with_start_length_errors_badarg(|mut process| {
        <BigInt as Num>::from_str_radix("576460752303423489", 10)
            .unwrap()
            .into_process(&mut process)
    });
}

#[test]
fn with_float_start_length_errors_badarg() {
    with_start_length_errors_badarg(|mut process| 1.0.into_process(&mut process));
}

#[test]
fn with_local_pid_start_length_errors_badarg() {
    with_start_length_errors_badarg(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_start_length_errors_badarg() {
    with_start_length_errors_badarg(|mut process| {
        Term::external_pid(1, 0, 0, &mut process).unwrap()
    });
}

#[test]
fn with_tuple_without_arity_2_errors_badarg() {
    with_start_length_errors_badarg(|mut process| Term::slice_to_tuple(&[], &mut process));
}

#[test]
fn with_map_errors_badarg() {
    with_start_length_errors_badarg(|mut process| Term::slice_to_map(&[], &mut process));
}

fn with_start_length_errors_badarg<S>(start_length: S)
where
    S: FnOnce(&mut Process) -> Term,
{
    errors_badarg(|mut process| {
        let binary = Term::slice_to_binary(&[], &mut process);
        let start_length = start_length(&mut process);

        erlang::binary_part_2(binary, start_length, &mut process)
    })
}
