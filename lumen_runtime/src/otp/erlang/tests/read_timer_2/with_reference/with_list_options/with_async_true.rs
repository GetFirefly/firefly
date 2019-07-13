use super::*;

mod with_timer;

#[test]
fn without_timer_returns_ok_and_sends_read_timer_message() {
    with_process(|process| {
        let timer_reference = process.next_reference().unwrap();

        assert_eq!(
            erlang::read_timer_2(timer_reference, options(process), process),
            Ok(atom_unchecked("ok"))
        );
        assert_eq!(
            receive_message(process),
            Some(read_timer_message(timer_reference, false.into(), process))
        );
    });
}

fn options(process: &ProcessControlBlock) -> Term {
    process
        .cons(async_option(true, process), Term::NIL)
        .unwrap()
}
