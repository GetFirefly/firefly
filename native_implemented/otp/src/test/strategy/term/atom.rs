use proptest::strategy::BoxedStrategy;

use firefly_rt::term::{Atom, Term};

pub fn is_not_encoding() -> BoxedStrategy<Term> {
    super::atom()
        .prop_filter("Encoding must not be latin1, unicode, utf8", |term| {
            let atom: Atom = (*term).try_into().unwrap();

            match atom.as_str() {
                "latin1" | "unicode" | "utf8" => false,
                _ => true,
            }
        })
        .boxed()
}
