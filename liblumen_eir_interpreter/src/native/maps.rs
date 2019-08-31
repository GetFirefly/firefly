use std::convert::{TryInto, AsRef};

use liblumen_alloc::erts::term::{atom_unchecked, Atom, Term, Boxed, Map};

use lumen_runtime::otp::maps;

use hashbrown::HashMap;

use crate::module::NativeModule;

pub fn make_maps() -> NativeModule {
    let mut native = NativeModule::new(Atom::try_from_str("maps").unwrap());

    native.add_simple(Atom::try_from_str("get").unwrap(), 2, |_proc, args| {
        let map_term: Boxed<Map> = args[1].try_into().unwrap();
        let hashmap_ref: &HashMap<Term, Term> = map_term.as_ref();

        Ok(hashmap_ref.get(&args[0]).unwrap().clone())
    });

    native.add_simple(Atom::try_from_str("find").unwrap(), 2, |proc, args| {
        let map_term: Boxed<Map> = args[1].try_into().unwrap();
        let hashmap_ref: &HashMap<Term, Term> = map_term.as_ref();

        let val = hashmap_ref.get(&args[0]).unwrap().clone();
        let ok = atom_unchecked("ok");

        let res = proc.tuple_from_slice(&[ok, val]).unwrap();
        Ok(res)
    });

    native.add_simple(Atom::try_from_str("merge").unwrap(), 2, |proc, args| {
        let res = maps::merge_2::native(proc, args[0], args[1]);
        Ok(res.unwrap())
    });

    native.add_simple(Atom::try_from_str("is_key").unwrap(), 2, |proc, args| {
        let res = maps::is_key_2::native(proc, args[0], args[1]);
        Ok(res.unwrap())
    });

    native
}
