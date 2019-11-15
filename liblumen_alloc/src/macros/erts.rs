#[macro_use]
mod exception;

#[macro_export]
macro_rules! atom {
    ($s:expr) => {
        $crate::erts::term::prelude::Atom::str_to_term($s)
    };
}

#[macro_export]
macro_rules! atom_from_str {
    ($s:expr) => {
        $crate::erts::term::prelude::Atom::try_from_str($s).unwrap()
    };
}

#[macro_export]
macro_rules! atom_from {
    ($e:expr) => {{
        #[allow(unused)]
        use core::convert::TryInto;
        let a: $crate::erts::term::prelude::Atom = ($e).try_into().unwrap();
        a
    }};
}

#[macro_export]
macro_rules! fixnum {
    ($num:expr) => {{
        #[allow(unused)]
        use $crate::erts::term::prelude::{Encode, Term};
        let t: Term = $crate::fixnum_from!($num).encode().unwrap();
        t
    }};
}

#[macro_export]
macro_rules! fixnum_from {
    ($num:expr) => {{
        #[allow(unused)]
        use core::convert::TryInto;
        let n: $crate::erts::term::prelude::SmallInteger = ($num).try_into().unwrap();
        n
    }};
}

#[macro_export]
macro_rules! cons {
    ($heap_or_proc:expr, $car:expr) => {
        ($heap_or_proc).cons($car, $crate::erts::term::prelude::Term::NIL).unwrap()
    };
    ($heap_or_proc:expr, $car:expr, $($cdr:expr),+) => {{
        let inner = $crate::cons!($heap_or_proc, $($cdr),*);
        ($heap_or_proc).cons($car, inner).unwrap()
    }}
}

#[macro_export]
macro_rules! improper_cons {
    ($heap_or_proc:expr, $tail:expr) => {
        compile_error!("improper lists can only be constructed from an even number of elements; possibly missing heap parameter?");
    };
    ($heap_or_proc:expr, $car:expr, $cdr:expr) => {
        ($heap_or_proc).cons($car, $cdr).unwrap()
    };
    ($heap_or_proc:expr, $car:expr, $cdr:expr, $($cadr:expr),+) => {{
        let inner = $crate::improper_cons!($heap_or_proc, $cdr, $($cadr),*);
        ($heap_or_proc).cons($car, inner).unwrap()
    }}
}
