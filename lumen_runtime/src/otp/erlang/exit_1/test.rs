use anyhow::*;

use proptest::prop_assert_eq;

use liblumen_alloc::exit;

use crate::otp::erlang::exit_1;
use crate::test::strategy;

#[test]
fn exits_with_reason() {
    run!(
        |arc_process| strategy::term(arc_process.clone()),
        |reason| {
            prop_assert_eq!(
                exit_1::native(reason),
                Err(exit!(reason, anyhow!("explicit exit from Erlang").into()).into())
            );

            Ok(())
        },
    );
}
