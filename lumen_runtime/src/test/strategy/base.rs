use proptest::strategy::{BoxedStrategy, Strategy};

pub fn base() -> BoxedStrategy<u8> {
    (2_u8..=36_u8).boxed()
}
