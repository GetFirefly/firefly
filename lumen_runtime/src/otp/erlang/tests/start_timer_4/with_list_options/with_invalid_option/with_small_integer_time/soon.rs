use super::*;

mod with_atom_destination;
mod with_local_pid_destination;

#[test]
fn with_small_integer_destination_errors_badarg() {
    with_destination_errors_badarg(|process| 0.into_process(&process));
}

#[test]
fn with_float_destination_errors_badarg() {
    with_destination_errors_badarg(|process| 1.0.into_process(&process));
}

#[test]
fn with_big_integer_destination_errors_badarg() {
    with_destination_errors_badarg(|process| (integer::small::MAX + 1).into_process(&process));
}

#[test]
fn with_local_reference_destination_errors_badarg() {
    with_destination_errors_badarg(|process| Term::next_local_reference(process));
}

#[test]
fn with_external_pid_destination_errors_badarg() {
    with_destination_errors_badarg(|process| Term::external_pid(1, 0, 0, &process).unwrap());
}

#[test]
fn with_tuple_destination_errors_badarg() {
    with_destination_errors_badarg(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_destination_errors_badarg() {
    with_destination_errors_badarg(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_empty_list_destination_errors_badarg() {
    with_destination_errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_destination_errors_badarg() {
    with_destination_errors_badarg(|process| list_term(&process));
}

#[test]
fn with_heap_binary_destination_errors_badarg() {
    with_destination_errors_badarg(|process| Term::slice_to_binary(&[1], &process));
}

#[test]
fn with_subbinary_destination_errors_badarg() {
    with_destination_errors_badarg(|process| {
        let original = Term::slice_to_binary(&[0, 1], &process);
        Term::subbinary(original, 1, 0, 1, 0, &process)
    });
}

fn milliseconds() -> u64 {
    timer::soon_milliseconds()
}

fn with_destination_errors_badarg<D>(destination: D)
where
    D: FnOnce(&Process) -> Term,
{
    with_process_arc(|process_arc| {
        let time = milliseconds().into_process(&process_arc);
        let message = Term::str_to_atom("message", DoNotCare).unwrap();
        let options = options(&process_arc);

        assert_badarg!(erlang::start_timer_4(
            time,
            destination(&process_arc),
            message,
            options,
            process_arc
        ));
    });
}
