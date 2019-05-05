use super::*;

mod with_atom_name;

#[test]
fn with_local_reference_name_errors_badarg() {
    with_name_errors_badarg(|process| Term::next_local_reference(process));
}

#[test]
fn with_empty_list_name_errors_badarg() {
    with_name_errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_name_errors_badarg() {
    with_name_errors_badarg(|process| {
        Term::cons(0.into_process(&process), 1.into_process(&process), &process)
    });
}

#[test]
fn with_small_integer_name_errors_badarg() {
    with_name_errors_badarg(|process| 0.into_process(&process));
}

#[test]
fn with_big_integer_name_errors_badarg() {
    with_name_errors_badarg(|process| (crate::integer::small::MAX + 1).into_process(&process));
}

#[test]
fn with_float_name_errors_badarg() {
    with_name_errors_badarg(|process| 0.0.into_process(&process));
}

#[test]
fn with_local_pid_name_errors_badarg() {
    with_name_errors_badarg(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_name_errors_badarg() {
    with_name_errors_badarg(|process| Term::external_pid(1, 2, 3, &process).unwrap());
}

#[test]
fn with_tuple_name_errors_badarg() {
    with_name_errors_badarg(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_name_errors_badarg() {
    with_name_errors_badarg(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_name_errors_badarg() {
    with_name_errors_badarg(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_name_errors_badarg() {
    with_name_errors_badarg(|process| bitstring!(1 :: 1, &process));
}

fn with_name_errors_badarg<N>(name: N)
where
    N: FnOnce(&Process) -> Term,
{
    with_process(|process| {
        let destination = Term::slice_to_tuple(&[name(process), erlang::node_0()], process);
        let message = Term::str_to_atom("message", DoNotCare).unwrap();

        assert_badarg!(erlang::send_2(destination, message, process));
    })
}
