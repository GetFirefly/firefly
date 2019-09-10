use liblumen_alloc::erts::term::{atom_unchecked, Atom};

use lumen_runtime::otp::maps;

use crate::module::NativeModule;

pub fn make_maps() -> NativeModule {
    let mut native = NativeModule::new(Atom::try_from_str("maps").unwrap());

    native.add_simple(Atom::try_from_str("merge").unwrap(), 2, |proc, args| {
        maps::merge_2::native(proc, args[0], args[1])
    });

    native.add_simple(Atom::try_from_str("is_key").unwrap(), 2, |proc, args| {
        maps::is_key_2::native(proc, args[0], args[1])
    });

    native.add_simple(Atom::try_from_str("find").unwrap(), 2, |proc, args| {
        maps::find_2::native(proc, args[0], args[1])
    });

    native.add_simple(Atom::try_from_str("get").unwrap(), 3, |proc, args| {
        maps::get_3::native(proc, args[0], args[1], args[2])
    });

    native.add_simple(Atom::try_from_str("get").unwrap(), 2, |proc, args| {
        maps::get_3::native(proc, args[0], args[1], atom_unchecked("nil"))
    });

    native
}
