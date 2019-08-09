use super::*;

use proptest::strategy::Strategy;

mod with_atom;
mod with_small_integer;

#[test]
fn without_atom_or_integer_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term(arc_process.clone())
                    .prop_filter("Unit must not be an atom or integer", |unit| {
                        !(unit.is_integer() || unit.is_atom())
                    }),
                |unit| {
                    prop_assert_eq!(
                        erlang::monotonic_time_1(unit, &arc_process),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn errors_badarg<U>(unit: U)
where
    U: FnOnce(&ProcessControlBlock) -> Term,
{
    super::errors_badarg(|process| erlang::monotonic_time_1(unit(&process), process));
}
