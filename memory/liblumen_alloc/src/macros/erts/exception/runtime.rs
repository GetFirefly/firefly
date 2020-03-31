#[macro_export]
macro_rules! badarg {
    ($source:expr) => {
        $crate::erts::exception::badarg(None, $source)
    };
    ($stacktrace:expr, $source:expr) => {
        $crate::erts::exception::badarg(Some($stacktrace), $source)
    };
}

#[macro_export]
macro_rules! undef {
    ($process:expr, $module:expr, $function:expr, $arguments:expr) => {
        $crate::erts::exception::undef(
            $process,
            $module,
            $function,
            $arguments,
            $crate::location!(),
            $crate::erts::term::prelude::Term::NIL,
        )
    };
    ($process:expr, $module:expr, $function:expr, $arguments:expr, $stacktrace_tail:expr) => {{
        $crate::erts::exception::undef(
            $process,
            $module,
            $function,
            $arguments,
            $crate::location!(),
            $stacktrace_tail,
        )
    }};
}

#[macro_export]
macro_rules! raise {
    ($class:expr, $reason:expr) => {
        $crate::erts::exception::raise($class, $reason, $crate::location!(), None)
    };
    ($class:expr, $reason:expr, $stacktrace:expr) => {
        $crate::erts::exception::raise($class, $reason, $crate::location!(), Some($stacktrace))
    };
}

#[macro_export]
macro_rules! error {
    ($reason:expr, $source:expr) => {
        $crate::erts::exception::error($reason, None, None, $source)
    };
    ($reason:expr, $arguments:expr, $source:expr) => {
        $crate::erts::exception::error($reason, Some($arguments), None, $source)
    };
}

#[macro_export]
macro_rules! exit {
    ($reason:expr, $source:expr) => {
        $crate::erts::exception::exit($reason, None, $source)
    };
    ($reason:expr, $stacktrace:expr, $source:expr) => {
        $crate::erts::exception::exit($reason, Some($stacktrace), $source)
    };
}

#[macro_export]
macro_rules! throw {
    ($reason:expr) => {
        $crate::erts::exception::throw($reason, $crate::location!(), None)
    };
    ($reason:expr, $stacktrace:expr) => {
        $crate::erts::exception::throw($reason, $crate::location!(), Some($stacktrace))
    };
}
