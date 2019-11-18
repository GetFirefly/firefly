#[macro_export]
macro_rules! alloc {
    () => {
        $crate::erts::exception::Alloc::new()
    };
}
