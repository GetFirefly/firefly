use super::*;

mod with_small_integer_time;

#[test]
fn with_float_time_errors_badarg() {
    with_time_errors_badarg(|process| 1.0.into_process(&process));
}

// BigInt is not tested because it would take too long and would always count as `long_term` for the
// super shot soon and later wheel sizes used for `cfg(test)`

#[test]
fn with_atom_time_errors_badarg() {
    with_time_errors_badarg(|_| Term::str_to_atom("atom", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_time_errors_badarg() {
    with_time_errors_badarg(|process| Term::local_reference(&process));
}

#[test]
fn with_local_pid_time_errors_badarg() {
    with_time_errors_badarg(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_time_errors_badarg() {
    with_time_errors_badarg(|process| Term::external_pid(1, 0, 0, &process).unwrap());
}

#[test]
fn with_tuple_time_errors_badarg() {
    with_time_errors_badarg(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_time_errors_badarg() {
    with_time_errors_badarg(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_empty_list_time_errors_badarg() {
    with_time_errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_time_errors_badarg() {
    with_time_errors_badarg(|process| list_term(&process));
}

#[test]
fn with_heap_binary_time_errors_badarg() {
    with_time_errors_badarg(|process| Term::slice_to_binary(&[1], &process));
}

#[test]
fn with_subbinary_time_errors_badarg() {
    with_time_errors_badarg(|process| bitstring!(1 :: 1, process));
}

fn options(_process: &Process) -> Term {
    Term::EMPTY_LIST
}

fn with_time_errors_badarg<T>(time: T)
where
    T: FnOnce(&Process) -> Term,
{
    with_process_arc(|process_arc| {
        let destination = process_arc.pid;
        let message = Term::str_to_atom("message", DoNotCare).unwrap();
        let options = options(&process_arc);

        assert_badarg!(erlang::start_timer_4(
            time(&process_arc),
            destination,
            message,
            options,
            process_arc
        ));
    });
}
