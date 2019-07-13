use super::*;

mod with_timer;

#[test]
fn without_timer_returns_false() {
    with_process(|process| {
        let timer_reference = process.next_reference().unwrap();

        assert_eq!(
            erlang::read_timer_1(timer_reference, process),
            Ok(false.into())
        );
    });
}
