use super::*;

mod with_timer;

#[test]
fn without_timer_returns_false() {
    with_process(|process| {
        let timer_reference = Term::next_local_reference(process);

        assert_eq!(
            erlang::read_timer_2(timer_reference, OPTIONS, process),
            Ok(false.into())
        );
    });
}
