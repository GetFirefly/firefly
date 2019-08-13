use proptest::strategy::{BoxedStrategy, Strategy};

pub fn node_id() -> BoxedStrategy<usize> {
    (1..=std::usize::MAX).boxed()
}
