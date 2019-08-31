use liblumen_alloc::erts::term::{atom_unchecked, Atom};

use lumen_runtime::otp::maps;

use crate::module::NativeModule;

pub fn make_maps() -> NativeModule {
    let mut native = NativeModule::new(Atom::try_from_str("maps").unwrap());

    native.add_simple(Atom::try_from_str("merge").unwrap(), 2, |proc, args| {
        let res = maps::merge_2::native(proc, args[0], args[1]);
        Ok(res.unwrap())
    });

    native.add_simple(Atom::try_from_str("is_key").unwrap(), 2, |proc, args| {
        let res = maps::is_key_2::native(proc, args[0], args[1]);
        Ok(res.unwrap())
    });

    native.add_simple(Atom::try_from_str("get").unwrap(), 3, |proc, args| {
        Ok(maps::get_3::native(proc, args[0], args[1], args[2]).unwrap())
    });

    native.add_simple(Atom::try_from_str("get").unwrap(), 2, |proc, args| {
        Ok(maps::get_3::native(proc, args[0], args[1], atom_unchecked("nil")).unwrap())
    });

    native
}
