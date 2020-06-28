use proptest::collection::SizeRange;
use proptest::strategy::{BoxedStrategy, Strategy};

pub fn with_size_range(size_range: SizeRange) -> BoxedStrategy<Vec<u8>> {
    proptest::collection::vec(proptest::prelude::any::<u8>(), size_range).boxed()
}
