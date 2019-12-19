macro_rules! term_try_into_atom {
    ($name:ident) => {
        crate::context::term_try_into_atom(stringify!($name), $name)
    };
}
