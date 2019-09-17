use super::*;

use liblumen_alloc::erts::term::atom_unchecked;

#[test]
fn without_true_or_false_is_false() {
    let term = atom_unchecked("atom");

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
