#[macro_export]
macro_rules! badarg {
    () => {
        $crate::erts::exception::runtime::Exception::badarg(file!(), line!(), column!())
    };
}

#[macro_export]
macro_rules! badarith {
    () => {{
        use $crate::erts::exception::runtime;

        runtime::Exception::badarith(file!(), line!(), column!())
    }};
}

#[macro_export]
macro_rules! badarity {
    ($process:expr, $function:expr, $arguments:expr) => {
        $crate::erts::exception::Exception::badarity(
            $process,
            $function,
            $arguments,
            file!(),
            line!(),
            column!(),
        )
    };
    ($process:expr, $function:expr, $arguments:expr,) => {
        $crate::badarity!($process, $function, $arguments)
    };
}

#[macro_export]
macro_rules! badfun {
    ($process:expr, $fun:expr) => {
        $crate::erts::exception::Exception::badfun($process, $fun, file!(), line!(), column!())
    };
}

#[macro_export]
macro_rules! error {
    ($reason:expr) => {{
        $crate::error!($reason, None)
    }};
    ($reason:expr, $arguments:expr) => {{
        use $crate::erts::exception::runtime::Class::Error;

        $crate::exception!(
            Error {
                arguments: $arguments
            },
            $reason
        )
    }};
    ($reason:expr, $arguments:expr,) => {{
        $crate::error!($reason, $arguments)
    }};
}

#[macro_export]
macro_rules! exception {
    ($class:expr, $reason:expr) => {
        $crate::exception!($class, $reason, None)
    };
    ($class:expr, $reason:expr,) => {
        $crate::exception!($class, $reason)
    };
    ($class:expr, $reason:expr, $stacktrace:expr) => {{
        use $crate::erts::exception::runtime;

        runtime::Exception {
            class: $class,
            reason: $reason,
            stacktrace: $stacktrace,
            file: file!(),
            line: line!(),
            column: column!(),
        }
    }};
    ($class:expr, $reason:expr, $stacktrace:expr) => {
        $crate::exception!($class, $reason, $stacktrace)
    };
}

#[macro_export]
macro_rules! exit {
    ($reason:expr) => {
        $crate::exit!($reason, None)
    };
    ($reason:expr, $stacktrace:expr) => {{
        use $crate::erts::exception::runtime::Class::Exit;

        $crate::exception!(Exit, $reason, $stacktrace)
    }};
}

#[macro_export]
macro_rules! raise {
    ($class:expr, $reason:expr, $stacktrace:expr) => {
        $crate::exception!($class, $reason, $stacktrace)
    };
    ($class:expr, $reason:expr, $stacktrace:expr,) => {
        $crate::exception!($class, $reason, $stacktrace)
    };
}

#[macro_export]
macro_rules! throw {
    ($reason:expr) => {{
        use $crate::exception::runtime::Class::Throw;

        $crate::exception!(Throw, $reason)
    }};
}
