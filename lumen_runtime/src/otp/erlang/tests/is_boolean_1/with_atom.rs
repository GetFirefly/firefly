use super::*;

#[test]
fn without_true_or_false_is_false() {
    let term = atom_unchecked("atom");

    assert_eq!(erlang::is_boolean_1(term), false.into());
}

#[test]
fn with_true_is_true() {
    let term = true.into();

    assert_eq!(erlang::is_boolean_1(term), true.into());
}

#[test]
fn with_false_is_true() {
    let term = false.into();

    assert_eq!(erlang::is_boolean_1(term), true.into());
}
