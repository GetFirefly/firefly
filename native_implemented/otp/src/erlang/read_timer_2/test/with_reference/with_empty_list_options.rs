use super::*;

mod with_timer;

#[test]
fn without_timer_returns_false() {
    with_process(|process| {
        let timer_reference = process.next_local_reference_term();

        assert_eq!(
            result(process, timer_reference, options(process)),
            Ok(false.into())
        );
    });
}

fn options(_: &Process) -> Term {
    Term::Nil
}
