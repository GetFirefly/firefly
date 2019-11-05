use super::*;

mod with_async_false;
mod with_async_true;
mod without_async;

#[test]
fn with_invalid_option() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term(arc_process.clone()), |option| {
                let timer_reference = arc_process.next_reference().unwrap();
                let options = arc_process.cons(option, Term::NIL).unwrap();

                prop_assert_eq!(
                    native(&arc_process, timer_reference, options),
                    Err(badarg!().into())
                );

                Ok(())
            })
            .unwrap();
    })
}

fn async_option(value: bool, process: &Process) -> Term {
    option("async", value, process)
}

fn info_option(value: bool, process: &Process) -> Term {
    option("info", value, process)
}

fn option(key: &str, value: bool, process: &Process) -> Term {
    process
        .tuple_from_slice(&[atom_unchecked(key), value.into()])
        .unwrap()
}
