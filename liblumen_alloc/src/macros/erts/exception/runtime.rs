#[macro_export]
macro_rules! badarg {
    ($process:expr) => {
        $crate::erts::exception::badarg($process, $crate::location!())
    };
}

#[macro_export]
macro_rules! badarith {
    ($process:expr) => {
        $crate::erts::exception::badarith($process, $crate::location!())
    };
}

#[macro_export]
macro_rules! badarity {
    ($process:expr, $function:expr, $arguments:expr) => {
        $crate::erts::exception::badarity($process, $function, $arguments, $crate::location!())
    };
}

#[macro_export]
macro_rules! badfun {
    ($process:expr, $fun:expr) => {
        $crate::erts::exception::badfun($process, $fun, $crate::location!())
    };
}

#[macro_export]
macro_rules! badkey {
    ($process:expr, $key:expr) => {
        $crate::erts::exception::badkey($process, $key, $crate::location!())
    };
}

#[macro_export]
macro_rules! badmap {
    ($process:expr, $map:expr) => {
        $crate::erts::exception::badmap($process, $map, $crate::location!())
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
    ($reason:expr) => {
        $crate::erts::exception::error($reason, None, $crate::location!(), None)
    };
    ($reason:expr, $arguments:expr) => {
        $crate::erts::exception::error($reason, Some($arguments), $crate::location!(), None)
    };
}

#[macro_export]
macro_rules! exit {
    ($reason:expr) => {
        $crate::erts::exception::exit($reason, $crate::location!(), None)
    };
    ($reason:expr, $stacktrace:expr) => {
        $crate::erts::exception::exit($reason, $crate::location!(), Some($stacktrace))
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
