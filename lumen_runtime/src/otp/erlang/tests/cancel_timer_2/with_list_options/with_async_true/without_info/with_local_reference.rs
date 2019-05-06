use super::*;

mod with_timer;

#[test]
fn without_timer_returns_ok_and_sends_cancel_timer_message() {
    with_process(|process| {
        let timer_reference = Term::next_local_reference(process);

        assert_eq!(
            erlang::cancel_timer_2(timer_reference, options(process), process),
            Ok(Term::str_to_atom("ok", DoNotCare).unwrap())
        );
        assert_eq!(
            receive_message(process),
            Some(cancel_timer_message(timer_reference, false.into(), process))
        );
    });
}
