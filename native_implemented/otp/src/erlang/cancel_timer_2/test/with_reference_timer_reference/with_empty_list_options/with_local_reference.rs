use super::*;

mod with_timer;

#[test]
fn without_timer_returns_false() {
    with_process(|process| {
        let timer_reference = process.next_reference().unwrap();
        let options = Term::NIL;

        assert_eq!(result(process, timer_reference, options), Ok(false.into()));
    });
}
