use super::*;

#[test]
fn with_small_integer_message_errors_badarg() {
    with_message_errors_badarg(|process| 0.into_process(process));
}

#[test]
fn with_float_message_errors_badarg() {
    with_message_errors_badarg(|process| 1.0.into_process(process));
}

#[test]
fn with_big_integer_message_errors_badarg() {
    with_message_errors_badarg(|process| (integer::small::MAX + 1).into_process(process));
}

#[test]
fn with_local_reference_message_errors_badarg() {
    with_message_errors_badarg(|process| Term::next_local_reference(process));
}

#[test]
fn with_external_pid_message_errors_badarg() {
    with_message_errors_badarg(|process| Term::external_pid(1, 0, 0, process).unwrap());
}

#[test]
fn with_tuple_message_errors_badarg() {
    with_message_errors_badarg(|process| Term::slice_to_tuple(&[], process));
}

#[test]
fn with_map_message_errors_badarg() {
    with_message_errors_badarg(|process| Term::slice_to_map(&[], process));
}

#[test]
fn with_empty_list_message_errors_badarg() {
    with_message_errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_message_errors_badarg() {
    with_message_errors_badarg(|process| list_term(process));
}

#[test]
fn with_heap_binary_message_errors_badarg() {
    with_message_errors_badarg(|process| Term::slice_to_binary(&[1], process));
}

#[test]
fn with_subbinary_message_errors_badarg() {
    with_message_errors_badarg(|process| bitstring!(1 :: 1, process));
}

fn with_message_errors_badarg<M>(message: M)
where
    M: FnOnce(&Process) -> Term,
{
    with_process_arc(|process_arc| {
        let destination_process_arc = process::local::new();
        let destination = registered_name();

        assert_eq!(
            erlang::register_2(
                destination,
                destination_process_arc.pid,
                process_arc.clone()
            ),
            Ok(true.into())
        );

        let milliseconds = milliseconds();
        let time = milliseconds.into_process(&process_arc);
        let message = message(&process_arc);
        let options = options(&process_arc);

        assert_badarg!(erlang::send_after_4(
            time,
            destination,
            message,
            options,
            process_arc.clone()
        ));
    })
}
