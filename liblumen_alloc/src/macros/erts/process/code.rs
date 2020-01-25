#[macro_export]
macro_rules! located_code {
    ($code:expr) => {
        $crate::erts::process::code::LocatedCode {
            code: $code,
            location: $crate::location!(),
        }
    };
}
