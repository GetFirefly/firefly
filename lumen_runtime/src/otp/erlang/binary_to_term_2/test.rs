mod with_safe;

use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{atom_unchecked, Term};

use crate::otp::erlang;
use crate::otp::erlang::binary_to_term_2::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
#[ignore]
fn with_used_with_binary_returns_how_many_bytes_were_consumed_along_with_term() {
    // <<131,100,0,5,"hello","world">>
    let byte_vec = vec![
        131, 100, 0, 5, 104, 101, 108, 108, 111, 119, 111, 114, 108, 100,
    ];

    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::binary::containing_bytes(byte_vec, arc_process.clone()),
                |binary| {
                    let options = options(&arc_process);
                    let term = atom_unchecked("hello");

                    prop_assert_eq!(
                        native(&arc_process, binary, options),
                        Ok(arc_process
                            .tuple_from_slice(&[term, arc_process.integer(9).unwrap()])
                            .unwrap())
                    );

                    // Using only `used` portion of binary returns the same result
                    let tuple = native(&arc_process, binary, options).unwrap();
                    let used_term =
                        erlang::element_2::native(arc_process.integer(2).unwrap(), tuple).unwrap();
                    let split_binary_tuple =
                        erlang::split_binary_2::native(&arc_process, binary, used_term).unwrap();
                    let prefix = erlang::element_2::native(
                        arc_process.integer(1).unwrap(),
                        split_binary_tuple,
                    )
                    .unwrap();

                    prop_assert_eq!(native(&arc_process, prefix, options), Ok(tuple));

                    // Without used returns only term

                    prop_assert_eq!(native(&arc_process, binary, Term::NIL), Ok(term));

                    Ok(())
                },
            )
            .unwrap();
    });

    fn options(process: &Process) -> Term {
        process.cons(atom_unchecked("used"), Term::NIL).unwrap()
    }
}
