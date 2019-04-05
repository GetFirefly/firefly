use super::*;

use std::sync::{Arc, RwLock};

use crate::environment::{self, Environment};

mod with_list;

#[test]
fn with_atom_errors_badarg() {
    errors_badarg(|_| Term::str_to_atom("iolist", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_errors_badarg() {
    errors_badarg(|mut process| Term::local_reference(&mut process));
}

#[test]
fn with_empty_list_returns_empty_binary() {
    with_process(|mut process| {
        assert_eq!(
            erlang::list_to_binary_1(Term::EMPTY_LIST, &mut process),
            Ok(Term::slice_to_binary(&[], &mut process))
        );
    });
}

#[test]
fn with_small_integer_errors_badarg() {
    errors_badarg(|mut process| 0.into_process(&mut process));
}

#[test]
fn with_big_integer_errors_badarg() {
    errors_badarg(|mut process| (crate::integer::small::MAX + 1).into_process(&mut process));
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
fn with_tuple_errors_badarg() {
    errors_badarg(|mut process| Term::slice_to_tuple(&[], &mut process));
}

#[test]
fn with_map_errors_badarg() {
    errors_badarg(|mut process| Term::slice_to_map(&[], &mut process));
}

#[test]
fn with_heap_binary_errors_badarg() {
    errors_badarg(|mut process| Term::slice_to_binary(&[], &mut process));
}

// > Bin1 = <<1,2,3>>.
// <<1,2,3>>
// > Bin2 = <<4,5>>.
// <<4,5>>
// > Bin3 = <<6>>.
// <<6>>
// > list_to_binary([Bin1,1,[2,3,Bin2],4|Bin3]).
// <<1,2,3,1,2,3,4,5,4,6>>
#[test]
fn otp_doctest_returns_binary() {
    with_process(|mut process| {
        let bin1 = Term::slice_to_binary(&[1, 2, 3], &mut process);
        let bin2 = Term::slice_to_binary(&[4, 5], &mut process);
        let bin3 = Term::slice_to_binary(&[6], &mut process);

        let iolist = Term::slice_to_improper_list(
            &[
                bin1,
                1.into_process(&mut process),
                Term::slice_to_list(
                    &[
                        2.into_process(&mut process),
                        3.into_process(&mut process),
                        bin2,
                    ],
                    &mut process,
                ),
                4.into_process(&mut process),
            ],
            bin3,
            &mut process,
        );

        assert_eq!(
            erlang::list_to_binary_1(iolist, &mut process),
            Ok(Term::slice_to_binary(
                &[1, 2, 3, 1, 2, 3, 4, 5, 4, 6],
                &mut process
            ))
        )
    });
}

#[test]
fn with_subbinary_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);

    assert_badarg!(erlang::list_to_binary_1(subbinary_term, &mut process));
}

fn errors_badarg<I>(iolist: I)
where
    I: FnOnce(&mut Process) -> Term,
{
    super::errors_badarg(|mut process| erlang::list_to_binary_1(iolist(&mut process), &mut process))
}
