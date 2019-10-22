use liblumen_alloc::erts::term::{AsTerm, Atom, Term};

pub const DEAD: &str = "nonode@nohost";

pub fn atom() -> Atom {
    Atom::try_from_str(DEAD).unwrap()
}

pub fn term() -> Term {
    unsafe { atom().as_term() }
}
