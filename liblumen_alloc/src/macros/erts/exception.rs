#[macro_use]
mod runtime;
#[macro_use]
mod system;

#[macro_export]
macro_rules! badkey {
    ($process:expr, $key:expr) => {{
        use $crate::erts::exception::Exception;

        Exception::badkey($process, $key, file!(), line!(), column!())
    }};
    ($process:expr, $key:expr,) => {{
        badkey!($process, $key)
    }};
}

#[macro_export]
macro_rules! badmap {
    ($process:expr, $map:expr) => {{
        use $crate::erts::exception::Exception;

        Exception::badmap($process, $map, file!(), line!(), column!())
    }};
    ($process:expr, $map:expr,) => {{
        badmap!($process, $map)
    }};
}

#[macro_export]
macro_rules! undef {
    ($process:expr, $module:expr, $function:expr, $arguments:expr) => {{
        use $crate::erts::term::prelude::Term;

        $crate::undef!($process, $module, $function, $arguments, Term::NIL)
    }};
    ($process:expr, $module:expr, $function:expr, $arguments:expr,) => {
        $crate::undef!($process, $module, $function, $arguments)
    };
    ($process:expr, $module:expr, $function:expr, $arguments:expr, $stacktrace_tail:expr) => {{
        use $crate::erts::exception::{runtime, Exception};

        Exception::undef(
            $process,
            $module,
            $function,
            $arguments,
            $stacktrace_tail,
            file!(),
            line!(),
            column!(),
        )
    }};
    ($process:expr, $module:expr, $function:expr, $arguments:expr, $stacktrace_tail:expr,) => {
        $crate::undef!($process, $module, $function, $arguments, $stacktrace_tail)
    };
}
