#[macro_use]
mod exception;

#[macro_export]
macro_rules! atom {
    ($s:expr) => {
        $crate::erts::term::prelude::Atom::str_to_term($s)
    }
}
