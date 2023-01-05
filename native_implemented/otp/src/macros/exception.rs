#[cfg(test)]
macro_rules! assert_badarg {
    ($actual:expr, $expected_substring:expr) => {{
        let actual = $actual;

        if let Err(ref non_null_erlang_exception) = actual
        {
            let erlang_exceptiion = unsafe { non_null_erlang_exception.as_ref() };
            assert_eq!(erlang_exceptiion.reason(), firefly_rt::term::atoms::BadArg.into());

            let source_message = format!("{:?}", error.source());
            let expected_substring = $expected_substring;

            assert!(
                source_message.contains(&expected_substring),
                "source message ({}) does not contain {:?}",
                source_message,
                &expected_substring
            );
        } else {
            panic!(
                "expected {} to error badarg, but got {:?}",
                stringify!($actual),
                actual
            );
        }
    }};
}

#[cfg(all(not(target_arch = "wasm32"), test))]
macro_rules! assert_badarith {
    ($left:expr) => {
        assert_error!(
            $left,
            liblumen_alloc::erts::term::prelude::Atom::str_to_term("badarith")
        )
    };
}

#[cfg(test)]
macro_rules! assert_error {
    ($left:expr, $reason:expr) => {{
        assert_eq!(
            $left,
            Err(error(
                $reason,
                None,
                Trace::capture(),
                Some(anyhow::anyhow!("Test").into())
            )
            .into())
        )
    }};
    ($left:expr, $reason:expr,) => {{
        assert_eq!($left, Err(error!($reason).into()))
    }};
    ($left:expr, $reason:expr, $arguments:expr) => {{
        assert_eq!($left, Err(error!($reason, $arguments).into()))
    }};
    ($left:expr, $reason:expr, $arguments:expr,) => {{
        assert_eq!($left, Err(error!($reason, $arguments).into()))
    }};
}

#[cfg(test)]
macro_rules! assert_has_message {
    ($process:expr, $message:expr) => {{
        let process: &liblumen_alloc::erts::process::Process = $process;

        assert!(
            has_message(process, $message),
            "Mailbox does not contain {:?} and instead contains {:?}",
            $message,
            process.mailbox.lock().borrow()
        );
    }};
}

#[cfg(test)]
macro_rules! prop_assert_error {
    ($actual:expr, $expected_error_name:literal, $expected_reason:expr, $expected_substring:expr $(,)?) => {{
        let actual = $actual;

        if let Err(liblumen_alloc::erts::exception::Exception::Runtime(
            liblumen_alloc::erts::exception::RuntimeException::Error(ref error),
        )) = actual
        {
            proptest::prop_assert_eq!(error.reason(), $expected_reason);

            let source_message = format!("{:?}", error.source());
            let expected_substring = $expected_substring;

            proptest::prop_assert!(
                source_message.contains(&expected_substring),
                "source message ({}) does not contain {:?}",
                source_message,
                &expected_substring
            );
        } else {
            return std::result::Result::Err(proptest::test_runner::TestCaseError::fail(format!(
                "expected {} to error {}, but got {:?}",
                $expected_error_name,
                stringify!($actual),
                actual
            )));
        }
    }};
}

#[cfg(test)]
macro_rules! prop_assert_badarg {
    ($actual:expr, $expected_substring:expr) => {{
        prop_assert_error!(
            $actual,
            "badarg",
            firefly_rt::term::atoms::Badarg,
            $expected_substring
        )
    }};
}

#[cfg(test)]
macro_rules! prop_assert_is_not_arity {
    ($actual:expr, $value:expr) => {
        prop_assert_is_not_type!($actual, "arity", $value, "an arity (an integer in 0-255)")
    };
}

#[cfg(test)]
macro_rules! prop_assert_is_not_atom {
    ($actual:expr, $name:ident) => {
        prop_assert_is_not_atom!($actual, stringify!($name), $name)
    };
    ($actual:expr, $name:expr, $value:expr) => {
        prop_assert_is_not_type!($actual, $name, $value, "an atom")
    };
}

#[cfg(test)]
macro_rules! prop_assert_is_not_binary {
    ($actual:expr, $name:ident) => {
        prop_assert_is_not_binary!($actual, stringify!($name), $name)
    };
    ($actual:expr, $name:expr, $value:expr) => {
        prop_assert_is_not_type!($actual, $name, $value, "a binary")
    };
}

#[cfg(test)]
macro_rules! prop_assert_is_not_boolean {
    ($actual:expr, $name:ident) => {
        prop_assert_is_not_boolean!($actual, stringify!($name), $name)
    };
    ($actual:expr, $name:expr, $value:expr) => {
        prop_assert_is_not_type!($actual, $name, $value, "a boolean")
    };
}

#[cfg(test)]
macro_rules! prop_assert_is_not_local_pid {
    ($actual:expr, $name:ident) => {
        prop_assert_is_not_local_pid!($actual, stringify!($name), $name)
    };
    ($actual:expr, $name:expr, $value:expr) => {
        prop_assert_is_not_type!($actual, $name, $value, "a pid")
    };
}

#[cfg(test)]
macro_rules! prop_assert_is_not_non_empty_list {
    ($actual:expr, $name:ident) => {
        prop_assert_is_not_non_empty_list!($actual, stringify!($name), $name)
    };
    ($actual:expr, $name:expr, $value:expr) => {
        prop_assert_is_not_type!($actual, $name, $value, "a non-empty list")
    };
}

#[cfg(test)]
macro_rules! prop_assert_is_not_number {
    ($actual:expr, $name:ident) => {
        prop_assert_is_not_number!($actual, stringify!($name), $name)
    };
    ($actual:expr, $name:expr, $value:expr) => {
        prop_assert_is_not_type!($actual, $name, $value, "a number (integer or float)")
    };
}

#[cfg(test)]
macro_rules! prop_assert_is_not_tuple {
    ($actual:expr, $name:ident) => {
        prop_assert_is_not_tuple!($actual, stringify!($name), $name)
    };
    ($actual:expr, $name:expr, $value:expr) => {
        prop_assert_is_not_type!($actual, $name, $value, "a tuple")
    };
}

#[cfg(test)]
macro_rules! prop_assert_is_not_type {
    ($actual:expr, $name:ident, $type:expr) => {
        prop_assert_is_not_type!($actual, stringify!($name), $name, $type)
    };
    ($actual:expr, $name:expr, $value:expr, $type:expr) => {
        prop_assert_badarg!($actual, format!("{} ({}) is not {}", $name, $value, $type))
    };
}

#[cfg(test)]
macro_rules! prop_assert_badarith {
    ($actual:expr, $expected_substring:expr) => {{
        prop_assert_error!(
            $actual,
            "badarith",
            firefly_rt::term::atoms::Badarith.into(),
            $expected_substring
        )
    }};
}

#[cfg(test)]
macro_rules! prop_assert_badkey {
    ($actual:expr, $process:expr, $expected_key:expr, $expected_substring:expr) => {{
        prop_assert_error!(
            $actual,
            "badkey",
            $process.tuple_term_from_term_slice(&[firefly_rt::term::atoms::Badkey.into(), $expected_key]),
            $expected_substring,
        )
    }};
}

#[cfg(test)]
macro_rules! prop_assert_badmap {
    ($actual:expr, $process:expr, $expected_map:ident) => {{
        prop_assert_badmap!(
            $actual,
            $process,
            $expected_map,
            format!(
                "{} ({}) is not a map",
                stringify!($expected_map),
                $expected_map
            ),
        )
    }};
    ($actual:expr, $process:expr, $expected_map:expr, $expected_substring:expr $(,)?) => {{
        prop_assert_error!(
            $actual,
            "badmap",
            $process.tuple_term_from_term_slice(&[firefly_rt::term::atoms::Badmap.into(), $expected_map]),
            $expected_substring,
        )
    }};
}
