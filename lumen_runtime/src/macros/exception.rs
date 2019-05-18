#[cfg(test)]
macro_rules! assert_badarg {
    ($left:expr) => {{
        use crate::atom::Existence::DoNotCare;
        use crate::term::Term;

        assert_error!($left, Term::str_to_atom("badarg", DoNotCare).unwrap())
    }};
}

#[cfg(test)]
macro_rules! assert_badarith {
    ($left:expr) => {{
        use crate::atom::Existence::DoNotCare;
        use crate::term::Term;

        assert_error!($left, Term::str_to_atom("badarith", DoNotCare).unwrap())
    }};
}

#[cfg(test)]
macro_rules! assert_badkey {
    ($left:expr, $key:expr, $process:expr) => {{
        use crate::atom::Existence::DoNotCare;
        use crate::term::Term;

        let badkey = Term::str_to_atom("badkey", DoNotCare).unwrap();
        let reason = Term::slice_to_tuple(&[badkey, $key], $process);

        assert_error!($left, reason)
    }};
    ($left:expr, $key:expr, $process:expr,) => {{
        assert_badkey!($left, $key, $process)
    }};
}

#[cfg(test)]
macro_rules! assert_badmap {
    ($left:expr, $map:expr, $process:expr) => {{
        use crate::atom::Existence::DoNotCare;
        use crate::term::Term;

        let badmap = Term::str_to_atom("badmap", DoNotCare).unwrap();
        let reason = Term::slice_to_tuple(&[badmap, $map], $process);

        assert_error!($left, reason)
    }};
    ($left:expr, $map:expr, $process:expr,) => {{
        assert_badmap($left, $map, $process)
    }};
}

#[cfg(test)]
macro_rules! assert_error {
    ($left:expr, $reason:expr) => {
        assert_eq!($left, Err(error!($reason)))
    };
    ($left:expr, $reason:expr,) => {
        assert_eq!($left, Err(error!($reason)))
    };
    ($left:expr, $reason:expr, $arguments:expr) => {
        assert_eq!($left, Err(error!($reason, $arguments)))
    };
    ($left:expr, $reason:expr, $arguments:expr,) => {
        assert_eq!($left, Err(error!($reason, $arguments)))
    };
}

#[cfg(test)]
macro_rules! assert_exit {
    ($left:expr, $reason:expr) => {
        assert_eq!($left, Err(exit!($reason)))
    };
    ($left:expr, $reason:expr,) => {
        assert_exit($left, $reason)
    };
}

#[cfg(test)]
macro_rules! assert_raises {
    ($left:expr, $class:expr, $reason:expr, $stacktrace:expr) => {
        assert_eq!($left, Err(raise!($class, $reason, $stacktrace)))
    };
    ($left:expr, $class:expr, $reason:expr, $stacktrace:expr,) => {
        assert_raises!($left, $class, $reason, $stacktrace)
    };
}

#[cfg(test)]
macro_rules! assert_throw {
    ($left:expr, $reason:expr) => {
        assert_eq!($left, Err(throw!($reason)))
    };
    ($left:expr, $reason:expr,) => {
        assert_throw($left, $reason)
    };
}

#[macro_export]
macro_rules! badarg {
    () => {{
        use $crate::atom::Existence::DoNotCare;
        use $crate::term::Term;

        $crate::error!(Term::str_to_atom("badarg", DoNotCare).unwrap())
    }};
}

macro_rules! badarith {
    () => {{
        use crate::atom::Existence::DoNotCare;
        use crate::term::Term;

        error!(Term::str_to_atom("badarith", DoNotCare).unwrap())
    }};
}

#[macro_export]
macro_rules! badarity {
    ($fun:expr, $arguments:expr, $process:expr) => {{
        use $crate::atom::Existence::DoNotCare;
        use $crate::term::Term;

        let badarity = Term::str_to_atom("badarity", DoNotCare).unwrap();
        let reason = Term::slice_to_tuple(
            &[
                badarity,
                Term::slice_to_tuple(&[$fun, $arguments], $process),
            ],
            $process,
        );

        $crate::error!(reason)
    }};
    ($fun:expr, $arguments:expr, $process:expr,) => {
        $crate::badarity!($fun, $arguments, $process);
    };
}

#[macro_export]
macro_rules! badfun {
    ($fun:expr, $process:expr) => {{
        use $crate::atom::Existence::DoNotCare;
        use $crate::term::Term;

        let badfun = Term::str_to_atom("badfun", DoNotCare).unwrap();
        let reason = Term::slice_to_tuple(&[badfun, $fun], $process);

        $crate::error!(reason)
    }};
}

macro_rules! badmap {
    ($map:expr, $process:expr) => {{
        use crate::atom::Existence::DoNotCare;
        use crate::term::Term;

        let badmap = Term::str_to_atom("badmap", DoNotCare).unwrap();
        let reason = Term::slice_to_tuple(&[badmap, $map], $process);

        error!(reason)
    }};
    ($map:expr, $process:expr,) => {{
        bad_map!($map, $process)
    }};
}

#[macro_export]
macro_rules! error {
    ($reason:expr) => {{
        $crate::error!($reason, None)
    }};
    ($reason:expr, $arguments:expr) => {{
        use $crate::exception::Class::Error;

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
        use $crate::exception::Exception;

        Exception {
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
        use crate::exception::Class::Exit;

        $crate::exception!(Exit, $reason, $stacktrace)
    }};
}

macro_rules! raise {
    ($class:expr, $reason:expr, $stacktrace:expr) => {
        exception!($class, $reason, $stacktrace)
    };
    ($class:expr, $reason:expr, $stacktrace:expr,) => {
        exception!($class, $reason, $stacktrace)
    };
}

macro_rules! throw {
    ($reason:expr) => {{
        use crate::exception::Class::Throw;

        exception!(Throw, $reason)
    }};
}

#[macro_export]
macro_rules! undef {
    ($module:expr, $function:expr, $arguments:expr, $process:expr) => {{
        use $crate::atom::Existence::DoNotCare;
        use $crate::term::Term;

        let undef = Term::str_to_atom("undef", DoNotCare).unwrap();
        let top = Term::slice_to_tuple(
            &[
                $module,
                $function,
                $arguments,
                // I'm not sure what this final empty list holds
                Term::EMPTY_LIST,
            ],
            $process,
        );
        let stacktrace = Term::cons(top, Term::EMPTY_LIST, $process);

        $crate::exit!(undef, Some(stacktrace))
    }};
    ($module:expr, $function:expr, $arguments:expr, $process:expr) => {
        $crate::undef!($module, $function, $arguments, $process)
    };
}
