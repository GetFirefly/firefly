use proptest::strategy::{BoxedStrategy, Strategy};

use crate::otp::erlang::tests::strategy::RANGE_INCLUSIVE;

pub fn strategy() -> BoxedStrategy<usize> {
    RANGE_INCLUSIVE.boxed()
}
