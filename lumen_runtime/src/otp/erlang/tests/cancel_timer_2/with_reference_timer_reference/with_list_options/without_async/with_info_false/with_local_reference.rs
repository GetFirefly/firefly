use super::*;

mod with_timer;

#[test]
fn without_timer_returns_ok() {
    with_process(|process| {
        let timer_reference = Term::next_local_reference(process);

        assert_eq!(
            erlang::cancel_timer_2(timer_reference, options(process), process),
            Ok(Term::str_to_atom("ok", DoNotCare).unwrap())
        );
    });
}
