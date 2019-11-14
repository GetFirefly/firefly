#[cfg(test)]
macro_rules! assert_badarg {
    ($left:expr, $stacktrace:expr) => {{
        assert_error!($left, $stacktrace, liblumen_alloc::atom!("badarg"))
    }};
}

#[cfg(all(not(target_arch = "wasm32"), test))]
macro_rules! assert_badarith {
    ($left:expr, $stacktrace:expr) => {{
        assert_error!($left, $stacktrace, liblumen_alloc::atom!("badarith"))
    }};
}

#[cfg(test)]
macro_rules! assert_error {
    ($left:expr, $stacktrace:expr, $reason:expr) => {{
        use liblumen_alloc::error;

        assert_eq!($left, Err(error!($stacktrace, $reason).into()))
    }};
    ($left:expr, $stacktrace:expr, $reason:expr,) => {{
        use liblumen_alloc::error;

        assert_eq!($left, Err(error!($stacktrace, $reason).into()))
    }};
    ($left:expr, $stacktrace:expr, $reason:expr, $arguments:expr) => {{
        use liblumen_alloc::error;

        assert_eq!($left, Err(error!($stacktrace, $reason, $arguments).into()))
    }};
    ($left:expr, $stacktrace:expr, $reason:expr, $arguments:expr,) => {{
        use liblumen_alloc::error;

        assert_eq!($left, Err(error!($stacktrace, $reason, $arguments).into()))
    }};
}
