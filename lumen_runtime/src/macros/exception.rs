#[cfg(test)]
macro_rules! assert_badarg {
    ($actual:expr, $expected_substring:expr) => {{
        let actual = $actual;

        if let Err(liblumen_alloc::erts::exception::Exception::Runtime(
            liblumen_alloc::erts::exception::RuntimeException::Error(ref error),
        )) = actual
        {
            assert_eq!(error.reason(), liblumen_alloc::atom!("badarg"));

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
        use liblumen_alloc::error;

        assert_eq!(
            $left,
            Err(error!($reason, anyhow::anyhow!("Test").into()).into())
        )
    }};
    ($left:expr, $reason:expr,) => {{
        use liblumen_alloc::error;

        assert_eq!($left, Err(error!($reason).into()))
    }};
    ($left:expr, $reason:expr, $arguments:expr) => {{
        use liblumen_alloc::error;

        assert_eq!($left, Err(error!($reason, $arguments).into()))
    }};
    ($left:expr, $reason:expr, $arguments:expr,) => {{
        use liblumen_alloc::error;

        assert_eq!($left, Err(error!($reason, $arguments).into()))
    }};
}

#[cfg(test)]
macro_rules! assert_is_not_non_empty_list {
    ($actual:expr, $name:ident) => {
        assert_is_not_non_empty_list!($actual, stringify!($name), $name)
    };
    ($actual:expr, $name:expr, $value:expr) => {
        assert_is_not_type!($actual, $name, $value, "a non-empty list")
    };
}

#[cfg(test)]
macro_rules! assert_is_not_type {
    ($actual:expr, $name:expr, $value:expr, $type:expr) => {
        assert_badarg!($actual, format!("{} ({}) is not {}", $name, $value, $type))
    };
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
            liblumen_alloc::atom!("badarg"),
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
macro_rules! prop_assert_is_not_integer {
    ($actual:expr, $name:ident) => {
        prop_assert_is_not_integer!($actual, stringify!($name), $name)
    };
    ($actual:expr, $name:expr, $value:expr) => {
        prop_assert_is_not_type!($actual, $name, $value, "an integer")
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
macro_rules! prop_assert_is_not_local_reference {
    ($actual:expr, $name:ident) => {
        prop_assert_is_not_local_reference!($actual, stringify!($name), $name)
    };
    ($actual:expr, $name:expr, $value:expr) => {
        prop_assert_is_not_type!($actual, $name, $value, "a reference")
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
macro_rules! prop_assert_is_not_non_negative_integer {
    ($actual:expr, $name:ident) => {
        prop_assert_is_not_non_negative_integer!($actual, stringify!($name), $name)
    };
    ($actual:expr, $name:expr, $value:expr) => {
        prop_assert_is_not_type!($actual, $name, $value, "a non-negative integer")
    };
}

#[cfg(test)]
macro_rules! prop_assert_is_not_time_unit {
    ($actual:expr, $name:ident) => {
        prop_assert_is_not_time_unit!($actual, stringify!($name), $name)
    };
    ($actual:expr, $name:expr, $value:expr) => {
        prop_assert_is_not_type!($actual, $name, $value, "a time unit")
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
            liblumen_alloc::atom!("badarith"),
            $expected_substring
        )
    }};
}

#[cfg(test)]
macro_rules! prop_assert_badarity {
    ($actual:expr, $process:expr, $fun:expr, $args:expr, $expected_substring:expr) => {{
        let process = $process;

        prop_assert_error!(
            $actual,
            "badarity",
            process
                .tuple_from_slice(&[
                    liblumen_alloc::atom!("badarity"),
                    process.tuple_from_slice(&[$fun, $args]).unwrap()
                ])
                .unwrap(),
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
            $process
                .tuple_from_slice(&[liblumen_alloc::atom!("badkey"), $expected_key])
                .unwrap(),
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
            $process
                .tuple_from_slice(&[liblumen_alloc::atom!("badmap"), $expected_map])
                .unwrap(),
            $expected_substring,
        )
    }};
}
