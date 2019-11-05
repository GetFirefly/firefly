use proptest::arbitrary::any;
use proptest::strategy::{BoxedStrategy, Strategy};

use liblumen_alloc::erts::term::Atom;

pub fn external() -> BoxedStrategy<Atom> {
    any::<String>()
        .prop_map(|s| Atom::try_from_str(&format!("{}@external", s)).unwrap())
        .boxed()
}
