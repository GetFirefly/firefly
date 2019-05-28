use proptest::collection::SizeRange;
use proptest::strategy::Strategy;

pub fn with_size_range(size_range: SizeRange) -> impl Strategy<Value = Vec<u8>> {
    proptest::collection::vec(proptest::prelude::any::<u8>(), size_range)
}
