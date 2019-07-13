#[cfg(test)]
macro_rules! assert_badarg {
    ($left:expr) => {{
        use liblumen_alloc::erts::term::atom_unchecked;

        assert_error!($left, atom_unchecked("badarg"))
    }};
}

#[cfg(all(not(target_arch = "wasm32"), test))]
macro_rules! assert_badarith {
    ($left:expr) => {{
        use liblumen_alloc::erts::term::atom_unchecked;

        assert_error!($left, atom_unchecked("badarith"))
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
