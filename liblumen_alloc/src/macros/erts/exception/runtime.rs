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
        $crate::badarity!($process, $process, $function, $arguments)
    };
    ($process:expr, $stacktrace:expr, $function:expr, $arguments:expr) => {
        $crate::erts::exception::badarity(
            $process,
            $stacktrace,
            $function,
            $arguments,
            $crate::location!(),
        )
    };
}

#[macro_export]
macro_rules! badfun {
    ($process:expr, $fun:expr) => {
        $crate::badfun!($process, $process, $fun)
    };
    ($process:expr, $stacktrace:expr, $fun:expr) => {
        $crate::erts::exception::badfun($process, $stacktrace, $fun, $crate::location!())
    };
}

#[macro_export]
macro_rules! badkey {
    ($process:expr, $key:expr) => {
        $crate::badkey!($process, $process, $key)
    };
    ($process:expr, $stacktrace:expr, $key:expr) => {
        $crate::erts::exception::badkey($process, $stacktrace, $key, $crate::location!())
    };
}

#[macro_export]
macro_rules! badmap {
    ($process:expr, $map:expr) => {
        $crate::badmap!($process, $process, $map)
    };
    ($process:expr, $stacktrace:expr, $map:expr) => {
        $crate::erts::exception::badmap($process, $stacktrace, $map, $crate::location!())
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
    ($stacktrace:expr, $reason:expr) => {
        $crate::erts::exception::error($reason, None, $crate::location!(), $stacktrace)
    };
    ($stacktrace:expr, $reason:expr, $arguments:expr) => {
        $crate::erts::exception::error($reason, Some($arguments), $crate::location!(), $stacktrace)
    };
}

#[macro_export]
macro_rules! exit {
    ($stacktrace:expr, $reason:expr) => {
        $crate::erts::exception::exit($reason, $crate::location!(), $stacktrace)
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
