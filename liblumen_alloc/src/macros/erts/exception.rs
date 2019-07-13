#[macro_use]
mod runtime;

#[macro_export]
macro_rules! badkey {
    ($process_control_block:expr, $key:expr) => {{
        use $crate::erts::exception::Exception;

        Exception::badkey(
            $process_control_block,
            $key,
            #[cfg(debug_assertions)]
            file!(),
            #[cfg(debug_assertions)]
            line!(),
            #[cfg(debug_assertions)]
            column!(),
        )
    }};
    ($process_control_block:expr, $key:expr,) => {{
        badkey!($process_control_block, $key)
    }};
}

#[macro_export]
macro_rules! badmap {
    ($process_control_block:expr, $map:expr) => {{
        use $crate::erts::exception::Exception;

        Exception::badmap(
            $process_control_block,
            $map,
            #[cfg(debug_assertions)]
            file!(),
            #[cfg(debug_assertions)]
            line!(),
            #[cfg(debug_assertions)]
            column!(),
        )
    }};
    ($process_control_block:expr, $map:expr,) => {{
        badmap!($process_control_block, $map)
    }};
}

#[macro_export]
macro_rules! undef {
    ($process_control_block:expr, $module:expr, $function:expr, $arguments:expr) => {{
        use $crate::erts::Term;

        $crate::undef!(
            $process_control_block,
            $module,
            $function,
            $arguments,
            Term::NIL
        )
    }};
    ($process_control_block:expr, $module:expr, $function:expr, $arguments:expr,) => {
        $crate::undef!($process_control_block, $module, $function, $arguments)
    };
    ($process_control_block:expr, $module:expr, $function:expr, $arguments:expr, $stacktrace_tail:expr) => {{
        use $crate::erts::exception::{runtime, Exception};

        Exception::undef(
            $process_control_block,
            $module,
            $function,
            $arguments,
            $stacktrace_tail,
            #[cfg(debug_assertions)]
            file!(),
            #[cfg(debug_assertions)]
            line!(),
            #[cfg(debug_assertions)]
            column!(),
        )
    }};
    ($process_control_block:expr, $module:expr, $function:expr, $arguments:expr, $stacktrace_tail:expr,) => {
        $crate::undef!(
            $process_control_block,
            $module,
            $function,
            $arguments,
            $stacktrace_tail
        )
    };
}
