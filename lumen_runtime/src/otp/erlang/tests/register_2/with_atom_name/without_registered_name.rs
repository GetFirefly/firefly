use super::*;

mod with_local_pid;

#[test]
fn with_atom_pid_or_process_errors_badarg() {
    with_pid_or_port_errors_badarg(|_| Term::str_to_atom("pid_or_port", DoNotCare).unwrap())
}

#[test]
fn with_empty_list_pid_or_process_errors_badarg() {
    with_pid_or_port_errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_pid_or_process_errors_badarg() {
    with_pid_or_port_errors_badarg(|process| {
        Term::cons(0.into_process(&process), 1.into_process(&process), &process)
    });
}

#[test]
fn with_small_integer_pid_or_process_errors_badarg() {
    with_pid_or_port_errors_badarg(|process| 0.into_process(&process));
}

#[test]
fn with_big_integer_pid_or_process_errors_badarg() {
    with_pid_or_port_errors_badarg(|process| {
        (crate::integer::small::MAX + 1).into_process(&process)
    });
}

#[test]
fn with_float_pid_or_process_errors_badarg() {
    with_pid_or_port_errors_badarg(|process| 0.0.into_process(&process));
}

#[test]
fn with_external_pid_pid_or_process_errors_badarg() {
    with_pid_or_port_errors_badarg(|process| Term::external_pid(1, 2, 3, &process).unwrap());
}

#[test]
fn with_tuple_pid_or_process_errors_badarg() {
    with_pid_or_port_errors_badarg(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_pid_or_process_errors_badarg() {
    with_pid_or_port_errors_badarg(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_pid_or_process_errors_badarg() {
    with_pid_or_port_errors_badarg(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_pid_or_process_errors_badarg() {
    with_pid_or_port_errors_badarg(|process| bitstring!(1 :: 1, &process));
}

fn with_pid_or_port_errors_badarg<P>(pid_or_port: P)
where
    P: FnOnce(&Process) -> Term,
{
    with_process_arc(|process_arc| {
        assert_badarg!(erlang::register_2(
            registered_name(),
            pid_or_port(&process_arc),
            process_arc
        ));
    })
}
