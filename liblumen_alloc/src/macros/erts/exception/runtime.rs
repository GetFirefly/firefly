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
        $crate::badarity!($process, $function, $arguments, $process)
    };
    ($process:expr, $function:expr, $arguments:expr, $stacktrace:expr) => {
        $crate::erts::exception::badarity(
            $process,
            $function,
            $arguments,
            $crate::location!(),
            $stacktrace,
        )
    };
}

#[macro_export]
macro_rules! badfun {
    ($process:expr, $fun:expr) => {
        $crate::badfun!($process, $fun, $process)
    };
    ($process:expr, $fun:expr, $stacktrace:expr) => {
        $crate::erts::exception::badfun($process, $fun, $crate::location!(), $stacktrace)
    };
}

#[macro_export]
macro_rules! badkey {
    ($process:expr, $key:expr) => {
        $crate::badkey!($process, $key, $process)
    };
    ($process:expr, $key:expr, $stacktrace:expr) => {
        $crate::erts::exception::badkey($process, $key, $crate::location!(), $stacktrace)
    };
}

#[macro_export]
macro_rules! badmap {
    ($process:expr, $map:expr) => {
        $crate::badmap!($process, $map, $process)
    };
    ($process:expr, $map:expr, $stacktrace:expr) => {
        $crate::erts::exception::badmap($process, $map, $crate::location!(), $stacktrace)
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
    ($stacktrace:expr, $class:expr, $reason:expr) => {
        $crate::erts::exception::raise($class, $reason, $crate::location!(), $stacktrace)
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
    ($stacktrace:expr, $reason:expr) => {
        $crate::erts::exception::throw($reason, $crate::location!(), $stacktrace)
    };
}
