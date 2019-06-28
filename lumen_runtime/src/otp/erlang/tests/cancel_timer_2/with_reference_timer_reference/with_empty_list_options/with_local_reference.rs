use super::*;

mod with_timer;

#[test]
fn without_timer_returns_false() {
    with_process(|process| {
        let timer_reference = Term::next_local_reference(process);
        let options = Term::EMPTY_LIST;

        assert_eq!(
            erlang::cancel_timer_2(timer_reference, options, process),
            Ok(false.into())
        );
    });
}
