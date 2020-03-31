use proptest::strategy::{BoxedStrategy, Strategy};

pub fn external() -> BoxedStrategy<usize> {
    super::id()
        .prop_filter("Can't be local node ID", |id| {
            id != &lumen_rt_full::distribution::nodes::node::id()
        })
        .boxed()
}
