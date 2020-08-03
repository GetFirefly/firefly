use liblumen_alloc::erts::term::prelude::Atom;

use crate::module::NativeModule;

macro_rules! trace {
    ($($t:tt)*) => (crate::runtime::sys::io::puts(&format_args!($($t)*).to_string()))
}

pub fn make_logger() -> NativeModule {
    let mut native = NativeModule::new(Atom::try_from_str("logger").unwrap());

    native.add_simple(Atom::try_from_str("allow").unwrap(), 2, |_proc, _args| {
        Ok(true.into())
    });

    native.add_simple(
        Atom::try_from_str("macro_log").unwrap(),
        4,
        |_proc, args| {
            trace!("{} {} {} {}", args[0], args[1], args[2], args[3]);
            Ok(true.into())
        },
    );

    native
}
