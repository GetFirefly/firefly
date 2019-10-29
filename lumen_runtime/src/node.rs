use liblumen_alloc::erts::term::prelude::*;

pub const DEAD: &str = "nonode@nohost";

pub fn atom() -> Atom {
    Atom::try_from_str(DEAD).unwrap()
}

pub fn term() -> Term {
    atom().encode().unwrap()
}
