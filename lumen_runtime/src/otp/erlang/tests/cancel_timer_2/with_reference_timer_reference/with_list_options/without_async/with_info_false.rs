use super::*;

mod with_local_reference;

#[test]
fn with_small_integer_timer_reference_errors_badarg() {
    with_timer_reference_errors_badarg(|process| process.integer(1).unwrap())
}

#[test]
fn with_float_timer_reference_errors_badarg() {
    with_timer_reference_errors_badarg(|process| process.float(1.0).unwrap());
}

#[test]
fn with_big_integer_timer_reference_errors_badarg() {
    with_timer_reference_errors_badarg(|process| {
        process.integer(SmallInteger::MAX_VALUE + 1).unwrap()
    });
}

#[test]
fn with_atom_timer_reference_errors_badarg() {
    with_timer_reference_errors_badarg(|_| atom_unchecked("atom"));
}

#[test]
fn with_local_pid_timer_reference_errors_badarg() {
    with_timer_reference_errors_badarg(|_| make_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_timer_reference_errors_badarg() {
    with_timer_reference_errors_badarg(|process| {
        process.external_pid_with_node_id(1, 0, 0).unwrap()
    });
}

#[test]
fn with_tuple_timer_reference_errors_badarg() {
    with_timer_reference_errors_badarg(|process| process.tuple_from_slice(&[]).unwrap());
}

#[test]
fn with_map_timer_reference_errors_badarg() {
    with_timer_reference_errors_badarg(|process| process.map_from_slice(&[]).unwrap());
}

#[test]
fn with_empty_list_timer_reference_errors_badarg() {
    with_timer_reference_errors_badarg(|_| Term::NIL);
}

#[test]
fn with_list_timer_reference_errors_badarg() {
    with_timer_reference_errors_badarg(|process| list_term(process));
}

#[test]
fn with_heap_binary_timer_reference_errors_badarg() {
    with_timer_reference_errors_badarg(|process| process.binary_from_bytes(&[1]).unwrap());
}

#[test]
fn with_subbinary_timer_reference_errors_badarg() {
    with_timer_reference_errors_badarg(|process| bitstring!(1 :: 1, process));
}

fn options(process: &Process) -> Term {
    process
        .cons(info_option(false, process), super::options(process))
        .unwrap()
}

fn with_timer_reference_errors_badarg<T>(timer_reference: T)
where
    T: FnOnce(&Process) -> Term,
{
    with_process(|process| {
        assert_badarg!(erlang::cancel_timer_2(
            timer_reference(process),
            options(process),
            process
        ));
    });
}
