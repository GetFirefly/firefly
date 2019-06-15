use super::*;

#[test]
fn with_atom_option_errors_badarg() {
    with_option_errors_badarg(|_| Term::str_to_atom("unsupported", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_option_errors_badarg() {
    with_option_errors_badarg(|process| Term::next_local_reference(process));
}

#[test]
fn with_small_integer_option_errors_badarg() {
    with_option_errors_badarg(|process| 0.into_process(&process));
}

#[test]
fn with_big_integer_option_errors_badarg() {
    with_option_errors_badarg(|process| (crate::integer::small::MAX + 1).into_process(&process));
}

#[test]
fn with_float_option_errors_badarg() {
    with_option_errors_badarg(|process| 0.0.into_process(&process));
}

#[test]
fn with_local_pid_option_errors_badarg() {
    with_option_errors_badarg(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_option_errors_badarg() {
    with_option_errors_badarg(|process| Term::external_pid(1, 2, 3, &process).unwrap());
}

#[test]
fn with_tuple_option_errors_badarg() {
    with_option_errors_badarg(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_option_errors_badarg() {
    with_option_errors_badarg(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_empty_list_option_errors_badarg() {
    with_option_errors_badarg(|process| {
        Term::cons(0.into_process(process), 1.into_process(process), process)
    });
}

#[test]
fn with_list_option_errors_badarg() {
    with_option_errors_badarg(|process| {
        Term::cons(0.into_process(process), 1.into_process(process), process)
    });
}

#[test]
fn with_heap_binary_option_errors_badarg() {
    with_option_errors_badarg(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_option_errors_badarg() {
    with_option_errors_badarg(|process| bitstring!(1 :: 1, &process));
}

fn with_option_errors_badarg<D>(option: D)
where
    D: FnOnce(&Process) -> Term,
{
    with_process(|process| {
        let destination = process.pid;
        let message = Term::str_to_atom("message", DoNotCare).unwrap();
        let options = Term::cons(option(process), Term::EMPTY_LIST, process);

        assert_badarg!(erlang::send_3(destination, message, options, process));
    });
}
