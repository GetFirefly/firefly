use proptest::strategy::{BoxedStrategy, Strategy};

pub fn node() -> BoxedStrategy<usize> {
    (1..=std::usize::MAX).boxed()
}
