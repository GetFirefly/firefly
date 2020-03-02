#[macro_export]
macro_rules! ok {
    () => {{
        ::liblumen_alloc::erts::term::prelude::Atom::from_str("ok")
            .encode()
            .unwrap()
    }};
}
