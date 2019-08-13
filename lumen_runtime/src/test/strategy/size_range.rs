use proptest::strategy::{BoxedStrategy, Strategy};

use crate::test::strategy::RANGE_INCLUSIVE;

pub fn strategy() -> BoxedStrategy<usize> {
    RANGE_INCLUSIVE.boxed()
}
