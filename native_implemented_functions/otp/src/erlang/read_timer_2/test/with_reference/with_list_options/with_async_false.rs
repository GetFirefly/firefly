use super::*;

mod with_timer;

#[test]
fn without_timer_returns_false() {
    with_process(|process| {
        let timer_reference = process.next_reference().unwrap();

        assert_eq!(
            result(process, timer_reference, options(process)),
            Ok(false.into())
        );
    });
}

fn options(process: &Process) -> Term {
    process
        .cons(async_option(false, process), Term::NIL)
        .unwrap()
}
