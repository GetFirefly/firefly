use std::convert::TryInto;

use proptest::strategy::{BoxedStrategy, Strategy};

use liblumen_alloc::erts::term::prelude::*;

pub fn is_not_encoding() -> BoxedStrategy<Term> {
    super::atom()
        .prop_filter("Encoding must not be latin1, unicode, utf8", |term| {
            let atom: Atom = (*term).try_into().unwrap();

            match atom.name() {
                "latin1" | "unicode" | "utf8" => false,
                _ => true,
            }
        })
        .boxed()
}
