use proptest::strategy::Strategy;

use crate::otp::erlang::tests::strategy::NON_EMPTY_RANGE_INCLUSIVE;

pub fn non_empty() -> impl Strategy<Value = usize> {
    NON_EMPTY_RANGE_INCLUSIVE.boxed()
}
