mod with_atom_process_identifier;
mod with_local_pid_process_identifier;
mod with_tuple_process_identifier;

use super::*;

#[test]
fn without_process_identifier_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &is_not_process_identifier(arc_process.clone()),
                |process_identifier| {
                    prop_assert_badarg!(
                        result(&arc_process, r#type(), process_identifier),
                        "process identifier must be `pid | registered_name() | {registered_name(), node()}`"
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn is_not_process_identifier(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![
        strategy::term::is_list(arc_process.clone()),
        strategy::term::local_reference(arc_process.clone()),
        strategy::term::is_function(arc_process.clone()),
        strategy::term::is_number(arc_process.clone()),
        strategy::term::is_bitstring(arc_process.clone())
    ]
    .boxed()
}

fn r#type() -> Term {
    Atom::str_to_term("process")
}
