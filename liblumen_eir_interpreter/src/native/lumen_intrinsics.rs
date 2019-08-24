use liblumen_alloc::erts::term::{atom_unchecked, Atom};

use crate::module::NativeModule;

pub fn make_lumen_intrinsics() -> NativeModule {
    let mut native = NativeModule::new(Atom::try_from_str("lumen_intrinsics").unwrap());

    native.add_simple(Atom::try_from_str("println").unwrap(), 1, |_proc, args| {
        assert!(args.len() == 1);
        lumen_runtime::system::io::puts(&format!("{}", args[0]));
        Ok(atom_unchecked("ok"))
    });

    native
}
