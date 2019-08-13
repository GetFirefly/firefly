use proptest::strategy::{BoxedStrategy, Strategy};

use crate::test::strategy::NON_EMPTY_RANGE_INCLUSIVE;

pub fn non_empty() -> BoxedStrategy<usize> {
    NON_EMPTY_RANGE_INCLUSIVE.boxed()
}
