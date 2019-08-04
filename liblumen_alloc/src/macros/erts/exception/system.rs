macro_rules! alloc {
    () => {
        $crate::erts::exception::system::Alloc {
            file: file!(),
            line: line!(),
            column: column!(),
        }
    };
}
