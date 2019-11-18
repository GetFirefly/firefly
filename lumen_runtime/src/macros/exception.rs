#[cfg(test)]
macro_rules! assert_badarg {
    ($left:expr) => {{
        use liblumen_alloc::erts::term::prelude::Atom;

        assert_error!($left, Atom::str_to_term("badarg"))
    }};
}

#[cfg(all(not(target_arch = "wasm32"), test))]
macro_rules! assert_badarith {
    ($left:expr) => {{
        use liblumen_alloc::erts::term::prelude::Atom;

        assert_error!($left, Atom::str_to_term("badarith"))
    }};
}

#[cfg(test)]
macro_rules! assert_error {
    ($left:expr, $reason:expr) => {{
        use liblumen_alloc::error;

        assert_eq!($left, Err(error!($reason).into()))
    }};
    ($left:expr, $reason:expr,) => {{
        use liblumen_alloc::error;

        assert_eq!($left, Err(error!($reason).into()))
    }};
    ($left:expr, $reason:expr, $arguments:expr) => {{
        use liblumen_alloc::error;

        assert_eq!($left, Err(error!($reason, $arguments).into()))
    }};
    ($left:expr, $reason:expr, $arguments:expr,) => {{
        use liblumen_alloc::error;

        assert_eq!($left, Err(error!($reason, $arguments).into()))
    }};
}
