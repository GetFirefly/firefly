use super::*;

use num_traits::Num;

#[test]
fn with_atom_errors_badarg() {
    errors_badarg(|_| Term::str_to_atom("atom", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_errors_badarg() {
    errors_badarg(|mut process| Term::local_reference(&mut process));
}

#[test]
fn with_empty_list_errors_badarg() {
    errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_errors_badarg() {
    errors_badarg(|mut process| list_term(&mut process));
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
fn with_tuple_returns_tuple_with_new_element_at_end() {
    with_process(|mut process| {
        let tuple_term = Term::slice_to_tuple(
            &[0.into_process(&mut process), 1.into_process(&mut process)],
            &mut process,
        );

        assert_eq!(
            erlang::append_element_2(tuple_term, 2.into_process(&mut process), &mut process),
            Ok(Term::slice_to_tuple(
                &[
                    0.into_process(&mut process),
                    1.into_process(&mut process),
                    2.into_process(&mut process)
                ],
                &mut process
            ))
        );
    });
}

#[test]
fn with_tuple_with_index_at_size_return_tuples_with_new_element_at_end() {
    with_process(|mut process| {
        let tuple = Term::slice_to_tuple(&[0.into_process(&mut process)], &mut process);

        assert_eq!(
            erlang::append_element_2(tuple, 1.into_process(&mut process), &mut process),
            Ok(Term::slice_to_tuple(
                &[0.into_process(&mut process), 1.into_process(&mut process)],
                &mut process
            ))
        );
    });
}

#[test]
fn with_map_errors_badarg() {
    errors_badarg(|mut process| Term::slice_to_map(&[], &mut process));
}

#[test]
fn with_heap_binary_errors_badarg() {
    errors_badarg(|mut process| Term::slice_to_binary(&[], &mut process));
}

#[test]
fn with_subbinary_errors_badarg() {
    errors_badarg(|mut process| {
        let original =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
        Term::subbinary(original, 0, 7, 2, 1, &mut process)
    });
}

fn errors_badarg<F>(tuple: F)
where
    F: FnOnce(&mut Process) -> Term,
{
    super::errors_badarg(|mut process| {
        erlang::append_element_2(
            tuple(&mut process),
            0.into_process(&mut process),
            &mut process,
        )
    });
}
