#[macro_export]
macro_rules! badarg {
    ($trace:expr) => {
        $crate::erts::exception::badarg($trace)
    };
}

#[macro_export]
macro_rules! badarg_with_source {
    ($source:expr) => {
        $crate::erts::exception::badarg_with_source($source)
    };
}

#[macro_export]
macro_rules! raise {
    ($class:expr, $reason:expr) => {
        let trace = crate::erts::process::trace::Trace::capture();
        $crate::erts::exception::raise($class, $reason, trace)
    };
}

#[macro_export]
macro_rules! raise_with_source {
    ($class:expr, $reason:expr, $source:expr) => {
        $crate::erts::exception::raise_with_source($class, $reason, $source)
    };
}

#[macro_export]
macro_rules! error {
    ($reason:expr, $trace:expr) => {
        $crate::erts::exception::error($reason, None, $trace)
    };
    ($reason:expr, $arguments:expr, $trace:expr) => {
        $crate::erts::exception::error($reason, Some($arguments), $trace)
    };
}

#[macro_export]
macro_rules! error_with_source {
    ($reason:expr, $source:expr) => {
        $crate::erts::exception::error_with_source($reason, None, $source)
    };
    ($reason:expr, $arguments:expr, $source:expr) => {
        $crate::erts::exception::error_with_source($reason, Some($arguments), $source)
    };
}

#[macro_export]
macro_rules! exit {
    ($reason:expr, $trace:expr) => {
        $crate::erts::exception::exit($reason, $trace)
    };
}

#[macro_export]
macro_rules! exit_with_source {
    ($reason:expr, $source:expr) => {
        $crate::erts::exception::exit_with_source($reason, $source)
    };
}

#[macro_export]
macro_rules! throw {
    ($reason:expr) => {
        let trace = crate::erts::process::trace::Trace::capture();
        $crate::erts::exception::throw($reason, trace)
    };
}

#[macro_export]
macro_rules! throw_with_source {
    ($reason:expr) => {
        $crate::erts::exception::throw($reason, $crate::location!())
    };
}
