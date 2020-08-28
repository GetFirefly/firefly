use anyhow::*;

use proptest::prop_assert_eq;

use liblumen_alloc::erts::exception::error;
use liblumen_alloc::erts::process::trace::Trace;

use crate::erlang::error_1::result;
use crate::test::strategy;

#[test]
fn errors_with_reason() {
    run!(
        |arc_process| strategy::term(arc_process.clone()),
        |reason| {
            prop_assert_eq!(
                result(reason),
                Err(error(reason, None, Trace::capture(), Some(anyhow!("test").into())).into())
            );

            Ok(())
        },
    );
}
