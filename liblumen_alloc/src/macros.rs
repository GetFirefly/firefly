#[macro_use]
mod erts;

#[macro_export]
macro_rules! location {
    () => {
        $crate::location::Location {
            file: file!(),
            line: line!(),
            column: column!(),
        }
    };
}
