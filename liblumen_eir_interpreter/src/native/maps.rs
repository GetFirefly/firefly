use liblumen_alloc::erts::term::atom_unchecked;
use liblumen_alloc::erts::term::Atom;

use lumen_runtime::otp::maps;

use crate::module::NativeModule;

pub fn make_maps() -> NativeModule {
    let mut native = NativeModule::new(Atom::try_from_str("maps").unwrap());

    native.add_simple(Atom::try_from_str("find").unwrap(), 2, |proc, args| {
        assert!(args.len() == 2);
        Ok(maps::find_2::native(proc, args[0], args[1]).unwrap())
    });

    native.add_simple(Atom::try_from_str("get").unwrap(), 3, |proc, args| {
        assert!(args.len() == 3);
        Ok(maps::get_3::native(proc, args[0], args[1], args[2]).unwrap())
    });

    native.add_simple(Atom::try_from_str("get").unwrap(), 2, |proc, args| {
        assert!(args.len() == 2);
        Ok(maps::get_3::native(proc, args[0], args[1], atom_unchecked("nil")).unwrap())
    });

    native
}
