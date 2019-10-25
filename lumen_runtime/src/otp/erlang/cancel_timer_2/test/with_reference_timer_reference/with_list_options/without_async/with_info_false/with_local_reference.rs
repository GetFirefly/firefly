use super::*;

mod with_timer;

#[test]
fn without_timer_returns_ok() {
    with_process(|process| {
        let timer_reference = process.next_reference().unwrap();

        assert_eq!(
            native(process, timer_reference, options(process)),
            Ok(Atom::str_to_term("ok"))
        );
    });
}
