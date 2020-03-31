use std::sync::Arc;

use proptest::strategy::Just;

use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Node;

use crate::runtime::distribution::nodes;

use crate::erlang::list_to_pid_1::native;
use crate::test::strategy;
use crate::test::with_process;

#[test]
fn without_list_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_list(arc_process.clone()),
            )
        },
        |(arc_process, string)| {
            prop_assert_is_not_non_empty_list!(native(&arc_process, string), string);

            Ok(())
        },
    );
}

#[test]
fn with_list_encoding_local_pid() {
    with_process(|process| {
        let string = process.charlist_from_str("<").unwrap();
        assert_badarg!(
            native(&process, string),
            format!("string ({}) is missing 'node.number.serial>'", string)
        );

        let string = process.charlist_from_str("<0").unwrap();
        assert_badarg!(
            native(&process, string),
            format!("string ({}) is missing '.number.serial>'", string)
        );

        let string = process.charlist_from_str("<0.").unwrap();
        assert_badarg!(
            native(&process, string),
            format!("string ({}) is missing 'number.serial>'", string)
        );

        let string = process.charlist_from_str("<0.1").unwrap();
        assert_badarg!(
            native(&process, string),
            format!("string ({}) is missing '.serial>", string)
        );

        let string = process.charlist_from_str("<0.1.").unwrap();
        assert_badarg!(
            native(&process, string),
            format!("string ({}) is missing 'serial>", string)
        );

        let string = process.charlist_from_str("<0.1.2").unwrap();
        assert_badarg!(
            native(&process, string),
            format!("string ({}) is missing '>'", string)
        );

        assert_eq!(
            native(&process, process.charlist_from_str("<0.1.2>").unwrap()),
            Ok(Pid::make_term(1, 2).unwrap())
        );

        assert_badarg!(
            native(&process, process.charlist_from_str("<0.1.2>?").unwrap()),
            "extra characters ([63]) beyond end of formatted pid"
        );
    })
}

#[test]
fn with_list_encoding_external_pid_without_known_node_errors_badarg() {
    with_process(|process| {
        let string = process.charlist_from_str("<").unwrap();
        assert_badarg!(
            native(&process, string),
            format!("string ({}) is missing 'node.number.serial>'", string)
        );

        let string = process.charlist_from_str("<2").unwrap();
        assert_badarg!(
            native(&process, string),
            format!("string ({}) is missing '.number.serial>'", string)
        );

        let string = process.charlist_from_str("<2.").unwrap();
        assert_badarg!(
            native(&process, string),
            format!("string ({}) is missing 'number.serial>'", string)
        );

        let string = process.charlist_from_str("<2.3").unwrap();
        assert_badarg!(
            native(&process, string),
            format!("string ({}) is missing '.serial>", string)
        );

        let string = process.charlist_from_str("<2.3.").unwrap();
        assert_badarg!(
            native(&process, string),
            format!("string ({}) is missing 'serial>", string)
        );

        let string = process.charlist_from_str("<2.3.").unwrap();
        assert_badarg!(
            native(&process, string),
            format!("string ({}) is missing 'serial>", string)
        );

        // MUST be a different `id` than other tests that insert the node.
        let arc_node = Arc::new(Node::new(2, Atom::try_from_str("2@external").unwrap(), 0));

        assert_badarg!(
            native(&process, process.charlist_from_str("<2.3.4>").unwrap()),
            "No node with id (2)"
        );

        nodes::insert(arc_node.clone());

        assert_eq!(
            native(&process, process.charlist_from_str("<2.3.4>").unwrap()),
            Ok(process.external_pid(arc_node, 3, 4).unwrap())
        );

        assert_badarg!(
            native(&process, process.charlist_from_str("<2.3.4>?").unwrap()),
            "extra characters ([63]) beyond end of formatted pid"
        );
    });
}
