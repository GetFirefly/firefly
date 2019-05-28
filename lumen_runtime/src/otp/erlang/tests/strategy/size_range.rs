use proptest::strategy::Strategy;

use crate::otp::erlang::tests::strategy::RANGE_INCLUSIVE;

pub fn strategy() -> impl Strategy<Value = usize> {
    RANGE_INCLUSIVE.boxed()
}
