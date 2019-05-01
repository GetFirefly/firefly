use super::*;

mod with_atom;

#[test]
fn with_small_integer_abs_errors_badarg() {
    with_abs_errors_badarg(|process| 1.into_process(process));
}

#[test]
fn with_float_abs_errors_badarg() {
    with_abs_errors_badarg(|process| 1.0.into_process(process));
}

#[test]
fn with_big_integer_errors_badarg() {
    with_abs_errors_badarg(|process| (integer::small::MAX + 1).into_process(process))
}

#[test]
fn with_local_reference_abs_errors_badarg() {
    with_abs_errors_badarg(|process| Term::local_reference(process));
}

#[test]
fn with_local_pid_abs_errors_badarg() {
    with_abs_errors_badarg(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_abs_errors_badarg() {
    with_abs_errors_badarg(|process| Term::external_pid(1, 0, 0, process).unwrap());
}

#[test]
fn with_tuple_abs_errors_badarg() {
    with_abs_errors_badarg(|process| Term::slice_to_tuple(&[], process));
}

#[test]
fn with_map_abs_errors_badarg() {
    with_abs_errors_badarg(|process| Term::slice_to_map(&[], process));
}

#[test]
fn with_empty_list_abs_errors_badarg() {
    with_abs_errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_abs_errors_badarg() {
    with_abs_errors_badarg(|process| list_term(process));
}

#[test]
fn with_heap_binary_abs_errors_badarg() {
    with_abs_errors_badarg(|process| Term::slice_to_binary(&[1], process));
}

#[test]
fn with_subbinary_abs_errors_badarg() {
    with_abs_errors_badarg(|process| bitstring!(1 :: 1, process));
}

fn options(abs: Term, process: &Process) -> Term {
    Term::cons(
        Term::slice_to_tuple(
            &[Term::str_to_atom("abs", DoNotCare).unwrap(), abs],
            process,
        ),
        Term::EMPTY_LIST,
        process,
    )
}

fn with_abs_errors_badarg<A>(abs: A)
where
    A: FnOnce(&Process) -> Term,
{
    with_process_arc(|process_arc| {
        let time = 1.into_process(&process_arc);
        let destination = process_arc.pid;
        let message = Term::str_to_atom("message", DoNotCare).unwrap();
        let options = options(abs(&process_arc), &process_arc);

        assert_badarg!(erlang::start_timer_4(
            time,
            destination,
            message,
            options,
            process_arc
        ));
    })
}
