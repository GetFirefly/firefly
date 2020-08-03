use super::*;

use proptest::prop_oneof;
use proptest::strategy::Strategy;

#[test]
fn without_list_or_bitstring_returns_false() {
    run!(
        |arc_process| {
            strategy::term(arc_process.clone())
                .prop_filter("Right cannot be a list or bitstring", |right| {
                    !(right.is_list() || right.is_bitstring())
                })
        },
        |right| {
            let left = Term::NIL;

            prop_assert_eq!(result(left, right), false.into());

            Ok(())
        },
    );
}

#[test]
fn with_list_or_bitstring_right_returns_true() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process()
                .prop_flat_map(|arc_process| list_or_bitstring(arc_process.clone())),
            |right| {
                let left = Term::NIL;

                prop_assert_eq!(result(left, right), true.into());

                Ok(())
            },
        )
        .unwrap();
}

fn list_or_bitstring(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![
        strategy::term::is_list(arc_process.clone()),
        strategy::term::is_bitstring(arc_process)
    ]
    .boxed()
}
