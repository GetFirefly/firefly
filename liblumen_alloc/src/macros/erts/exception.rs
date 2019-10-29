#[macro_use]
mod runtime;
#[macro_use]
mod system;

#[macro_export]
macro_rules! location {
    () => {
        $crate::erts::exception::Location::new(file!(), line!(), column!())
    }
}
