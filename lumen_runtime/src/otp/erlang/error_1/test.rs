use anyhow::*;

use proptest::prop_assert_eq;

use liblumen_alloc::error;

use crate::otp::erlang::error_1::native;
use crate::test::{run, strategy};

#[test]
fn errors_with_reason() {
    run(
        file!(),
        |arc_process| strategy::term(arc_process.clone()),
        |reason| {
            prop_assert_eq!(
                native(reason),
                Err(error!(reason, anyhow!("test").into()).into())
            );

            Ok(())
        },
    );
}
