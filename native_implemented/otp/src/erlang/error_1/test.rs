use anyhow::*;

use proptest::prop_assert_eq;

use liblumen_alloc::error;

use crate::erlang::error_1::result;
use crate::test::strategy;

#[test]
fn errors_with_reason() {
    run!(
        |arc_process| strategy::term(arc_process.clone()),
        |reason| {
            prop_assert_eq!(
                result(reason),
                Err(error!(reason, anyhow!("test").into()).into())
            );

            Ok(())
        },
    );
}
