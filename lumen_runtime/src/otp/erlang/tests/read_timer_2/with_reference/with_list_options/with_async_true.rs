use super::*;

mod with_timer;

#[test]
fn without_timer_returns_ok_and_sends_read_timer_message() {
    with_process(|process| {
        let timer_reference = Term::next_local_reference(process);

        assert_eq!(
            erlang::read_timer_2(timer_reference, options(process), process),
            Ok(Term::str_to_atom("ok", DoNotCare).unwrap())
        );
        assert_eq!(
            receive_message(process),
            Some(read_timer_message(timer_reference, false.into(), process))
        );
    });
}

fn options(process: &Process) -> Term {
    Term::cons(async_option(true, process), Term::EMPTY_LIST, process)
}
