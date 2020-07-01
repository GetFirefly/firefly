use liblumen_alloc::erts::term::prelude::*;

use crate::module::NativeModule;

pub fn make_lumen_intrinsics() -> NativeModule {
    let mut native = NativeModule::new(Atom::try_from_str("lumen_intrinsics").unwrap());

    native.add_simple(Atom::try_from_str("println").unwrap(), 1, |_proc, args| {
        crate::runtime::sys::io::puts(&format!("{}", args[0]));
        Ok(Atom::str_to_term("ok"))
    });

    native.add_simple(Atom::try_from_str("format").unwrap(), 1, |proc, args| {
        let string = format!("{}", args[0]);
        let term = proc.binary_from_str(&string).unwrap();
        Ok(term)
    });

    native.add_simple(
        Atom::try_from_str("dump_process_heap").unwrap(),
        0,
        |proc, _args| {
            crate::runtime::sys::io::puts(&format!("{:?}", proc.acquire_heap()));
            Ok(Atom::str_to_term("ok"))
        },
    );

    native
}
