#[macro_export]
macro_rules! badarg {
    () => {{
        use $crate::erts::exception::runtime;

        runtime::Exception::badarg(
            #[cfg(debug_assertions)]
            file!(),
            #[cfg(debug_assertions)]
            line!(),
            #[cfg(debug_assertions)]
            column!(),
        )
    }};
}

#[macro_export]
macro_rules! badarith {
    () => {{
        use $crate::erts::exception::runtime;

        runtime::Exception::badarith(
            #[cfg(debug_assertions)]
            file!(),
            #[cfg(debug_assertions)]
            line!(),
            #[cfg(debug_assertions)]
            column!(),
        )
    }};
}

#[macro_export]
macro_rules! badarity {
    ($process:expr, $function:expr, $arguments:expr) => {{
        $crate::erts::exception::Exception::badarity(
            $process,
            $function,
            $arguments,
            #[cfg(debug_assertions)]
            file!(),
            #[cfg(debug_assertions)]
            line!(),
            #[cfg(debug_assertions)]
            column!(),
        )
    }};
    ($process:expr, $function:expr, $arguments:expr,) => {{
        $crate::badarity!($function, $arguments, $process)
    }};
}

#[macro_export]
macro_rules! badfun {
    ($process:expr, $fun:expr) => {{
        use liblumen_alloc::{atom_unchecked, Term};

        let badfun = atom_unchecked("badfun");
        let reason = Term::slice_to_tuple(&[badfun, $fun], $process);

        $crate::error!(reason)
    }};
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
            #[cfg(debug_assertions)]
            file: file!(),
            #[cfg(debug_assertions)]
            line: line!(),
            #[cfg(debug_assertions)]
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
