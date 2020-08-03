use liblumen_alloc::erts::term::prelude::*;

use liblumen_otp::maps;

use crate::module::NativeModule;

pub fn make_maps() -> NativeModule {
    let mut native = NativeModule::new(Atom::try_from_str("maps").unwrap());

    native.add_simple(Atom::try_from_str("find").unwrap(), 2, |proc, args| {
        maps::find_2::result(proc, args[0], args[1])
    });

    native.add_simple(Atom::try_from_str("from_list").unwrap(), 1, |proc, args| {
        maps::from_list_1::result(proc, args[0])
    });

    native.add_simple(Atom::try_from_str("get").unwrap(), 2, |proc, args| {
        maps::get_3::result(proc, args[0], args[1], Atom::str_to_term("nil"))
    });

    native.add_simple(Atom::try_from_str("get").unwrap(), 3, |proc, args| {
        maps::get_3::result(proc, args[0], args[1], args[2])
    });

    native.add_simple(Atom::try_from_str("is_key").unwrap(), 2, |proc, args| {
        maps::is_key_2::result(proc, args[0], args[1])
    });

    native.add_simple(Atom::try_from_str("keys").unwrap(), 1, |proc, args| {
        maps::keys_1::result(proc, args[0])
    });

    native.add_simple(Atom::try_from_str("merge").unwrap(), 2, |proc, args| {
        maps::merge_2::result(proc, args[0], args[1])
    });

    native.add_simple(Atom::try_from_str("put").unwrap(), 3, |proc, args| {
        maps::put_3::result(proc, args[0], args[1], args[2])
    });

    native.add_simple(Atom::try_from_str("remove").unwrap(), 2, |proc, args| {
        maps::remove_2::result(proc, args[0], args[1])
    });

    native.add_simple(Atom::try_from_str("take").unwrap(), 2, |proc, args| {
        maps::take_2::result(proc, args[0], args[1])
    });

    native.add_simple(Atom::try_from_str("update").unwrap(), 3, |proc, args| {
        maps::update_3::result(proc, args[0], args[1], args[2])
    });

    native.add_simple(Atom::try_from_str("values").unwrap(), 1, |proc, args| {
        maps::values_1::result(proc, args[0])
    });

    native
}
