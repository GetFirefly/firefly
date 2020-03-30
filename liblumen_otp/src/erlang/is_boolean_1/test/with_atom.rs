use super::*;

use liblumen_alloc::erts::term::prelude::Atom;

#[test]
fn without_true_or_false_is_false() {
    let term = Atom::str_to_term("atom");

    assert_eq!(native(term), false.into());
}

#[test]
fn with_true_is_true() {
    let term = true.into();

    assert_eq!(native(term), true.into());
}

#[test]
fn with_false_is_true() {
    let term = false.into();

    assert_eq!(native(term), true.into());
}
