use super::*;

mod with_timer;

#[test]
fn without_timer_returns_false() {
    with_process(|process| {
        let timer_reference = Term::next_local_reference(process);

        assert_eq!(
            erlang::cancel_timer_1(timer_reference, process),
            Ok(false.into())
        );
    });
}
