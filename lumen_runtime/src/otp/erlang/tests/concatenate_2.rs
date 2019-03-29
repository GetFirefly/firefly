use super::*;

use num_traits::Num;

mod with_empty_list;
mod with_list;

#[test]
fn with_atom_errors_badarg() {
    errors_badarg(|_| {
        let list = Term::str_to_atom("list", DoNotCare).unwrap();
        let term = Term::str_to_atom("term", DoNotCare).unwrap();

        (list, term)
    });
}

#[test]
fn with_local_reference_errors_badarg() {
    errors_badarg(|mut process| {
        let list = Term::local_reference(&mut process);
        let term = Term::local_reference(&mut process);

        (list, term)
    });
}

#[test]
fn with_improper_list_errors_badarg() {
    errors_badarg(|mut process| {
        let list = Term::cons(
            0.into_process(&mut process),
            1.into_process(&mut process),
            &mut process,
        );
        let term = Term::cons(
            2.into_process(&mut process),
            3.into_process(&mut process),
            &mut process,
        );

        (list, term)
    });
}

#[test]
fn with_small_integer_errors_badarg() {
    errors_badarg(|mut process| {
        let list = 0.into_process(&mut process);
        let term = 1.into_process(&mut process);

        (list, term)
    });
}

#[test]
fn with_big_integer_errors_badarg() {
    errors_badarg(|mut process| {
        let list = <BigInt as Num>::from_str_radix("576460752303423489", 10)
            .unwrap()
            .into_process(&mut process);
        let term = <BigInt as Num>::from_str_radix("576460752303423490", 10)
            .unwrap()
            .into_process(&mut process);

        (list, term)
    });
}

#[test]
fn with_float_errors_badarg() {
    errors_badarg(|mut process| {
        let list = 1.0.into_process(&mut process);
        let term = 2.0.into_process(&mut process);

        (list, term)
    });
}

#[test]
fn with_local_pid_errors_badarg() {
    errors_badarg(|_| {
        let list = Term::local_pid(0, 1).unwrap();
        let term = Term::local_pid(1, 2).unwrap();

        (list, term)
    });
}

#[test]
fn with_external_pid_errors_badarg() {
    errors_badarg(|mut process| {
        let list = Term::external_pid(1, 2, 3, &mut process).unwrap();
        let term = Term::external_pid(4, 5, 6, &mut process).unwrap();

        (list, term)
    });
}

#[test]
fn with_tuple_errors_badarg() {
    errors_badarg(|mut process| {
        let list = Term::slice_to_tuple(&[], &mut process);
        let term = Term::slice_to_tuple(&[], &mut process);

        (list, term)
    });
}

#[test]
fn with_map_is_errors_badarg() {
    errors_badarg(|mut process| {
        let list = Term::slice_to_map(&[], &mut process);
        let term = Term::slice_to_map(&[], &mut process);

        (list, term)
    });
}

#[test]
fn with_heap_binary_errors_badarg() {
    errors_badarg(|mut process| {
        let list = Term::slice_to_binary(&[], &mut process);
        let term = Term::slice_to_binary(&[], &mut process);

        (list, term)
    });
}

#[test]
fn with_subbinary_errors_badarg() {
    errors_badarg(|mut process| {
        let binary_term =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
        let list = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
        let term = Term::subbinary(binary_term, 0, 7, 2, 0, &mut process);

        (list, term)
    });
}

fn errors_badarg<F>(list_term: F)
where
    F: FnOnce(&mut Process) -> (Term, Term),
{
    super::errors_badarg(|mut process| {
        let (list, term) = list_term(&mut process);

        erlang::concatenate_2(list, term, &mut process)
    });
}
