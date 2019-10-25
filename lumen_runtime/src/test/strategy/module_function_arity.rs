use proptest::arbitrary::any;
use proptest::strategy::{BoxedStrategy, Strategy};

use liblumen_alloc::erts::term::prelude::Atom;

use crate::test::strategy::term::NON_EXISTENT_ATOM_PREFIX;

pub fn module() -> BoxedStrategy<Atom> {
    atom()
}

pub fn function() -> BoxedStrategy<Atom> {
    atom()
}

pub fn arity() -> BoxedStrategy<u8> {
    (0_u8..=255_u8).boxed()
}

fn atom() -> BoxedStrategy<Atom> {
    any::<String>()
        .prop_filter("Reserved for existing/safe atom tests", |s| {
            !s.starts_with(NON_EXISTENT_ATOM_PREFIX)
        })
        .prop_map(|s| Atom::try_from_str(&s).unwrap())
        .boxed()
}
