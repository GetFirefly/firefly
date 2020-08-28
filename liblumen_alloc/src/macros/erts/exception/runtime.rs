#[macro_export]
macro_rules! badarg {
    ($trace:expr) => {
        $crate::erts::exception::badarg($trace, None)
    };
    ($trace:expr, $source:expr) => {
        $crate::erts::exception::badarg($trace, Some($source))
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
        $crate::erts::exception::error($reason, None, $trace, None)
    };
    ($reason:expr, $arguments:expr, $trace:expr) => {
        $crate::erts::exception::error($reason, Some($arguments), $trace, None)
    };
    ($reason:expr, $arguments:expr, $trace:expr, $source:expr) => {
        $crate::erts::exception::error($reason, Some($arguments), $trace, Some($source))
    };
}

#[macro_export]
macro_rules! exit {
    ($reason:expr, $trace:expr) => {
        $crate::erts::exception::exit($reason, $trace, None)
    };
    ($reason:expr, $trace:expr, $source:expr) => {
        $crate::erts::exception::exit($reason, $trace, Some($source))
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
