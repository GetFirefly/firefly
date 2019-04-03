use super::*;

use num_traits::Num;

//use crate::exception::Class;

mod with_atom_class;

#[test]
fn with_local_reference_class_errors_badarg() {
    with_class_errors_badarg(|mut process| Term::local_reference(&mut process));
}

#[test]
fn with_empty_list_class_errors_badarg() {
    with_class_errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_class_errors_badarg() {
    with_class_errors_badarg(|mut process| list_term(&mut process));
}

#[test]
fn with_small_integer_class_errors_badarg() {
    with_class_errors_badarg(|mut process| 0usize.into_process(&mut process));
}

#[test]
fn with_big_integer_class_errors_badarg() {
    with_class_errors_badarg(|mut process| {
        <BigInt as Num>::from_str_radix("576460752303423489", 10)
            .unwrap()
            .into_process(&mut process)
    });
}

#[test]
fn with_float_class_errors_badarg() {
    with_class_errors_badarg(|mut process| 1.0.into_process(&mut process));
}

#[test]
fn with_local_pid_class_errors_badarg() {
    with_class_errors_badarg(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_class_errors_badarg() {
    with_class_errors_badarg(|mut process| Term::external_pid(1, 0, 0, &mut process).unwrap());
}

#[test]
fn with_tuple_class_errors_badarg() {
    with_class_errors_badarg(|mut process| Term::slice_to_tuple(&[], &mut process));
}

#[test]
fn with_map_class_errors_badarg() {
    with_class_errors_badarg(|mut process| Term::slice_to_map(&[], &mut process));
}

#[test]
fn with_heap_binary_class_errors_badarg() {
    with_class_errors_badarg(|mut process| {
        // :erlang.term_to_binary(:atom)
        Term::slice_to_binary(&[131, 100, 0, 4, 97, 116, 111, 109], &mut process)
    });
}

#[test]
fn with_subbinary_class_errors_badarg() {
    with_class_errors_badarg(|mut process| {
        // <<1::1, :erlang.term_to_binary(:atom) :: binary>>
        let original_term = Term::slice_to_binary(
            &[193, 178, 0, 2, 48, 186, 55, 182, 0b1000_0000],
            &mut process,
        );
        Term::subbinary(original_term, 0, 1, 8, 0, &mut process)
    });
}

fn with_class_errors_badarg<C>(class: C)
where
    C: FnOnce(&mut Process) -> Term,
{
    with_process(|mut process| {
        let class = class(&mut process);
        let reason = Term::str_to_atom("reason", DoNotCare).unwrap();
        let stacktrace = Term::EMPTY_LIST;

        assert_badarg!(erlang::raise_3(class, reason, stacktrace));
    })
}

//fn raises<A>(class_reason_stacktrace: A)
//    where
//        A: FnOnce(&mut Process) -> ((Term, Class), Term, Term),
//{
//    with_process(|mut process| {
//        let ((class_term, class_class), reason, stacktrace) = class_reason_stacktrace(&mut
// process);
//
//        assert_raises!(erlang::raise_3(class_term, reason, stacktrace), class_class, reason,
// Some(stacktrace));    })
//}
