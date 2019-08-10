use liblumen_alloc::erts::term::{atom_unchecked, AsTerm, Atom, Closure, Integer, Term, TypedTerm};
use lumen_runtime::otp::erlang;

use crate::module::NativeModule;

pub fn make_erlang() -> NativeModule {
    let mut native = NativeModule::new(Atom::try_from_str("erlang").unwrap());

    native.add_fun(Atom::try_from_str("<").unwrap(), 2, |proc, args| {
        assert!(args.len() == 2);
        Ok(erlang::is_less_than_2(args[0], args[1]))
    });

    native.add_fun(Atom::try_from_str("-").unwrap(), 2, |proc, args| {
        assert!(args.len() == 2);
        Ok(erlang::subtract_2(args[0], args[1], proc).unwrap())
    });

    native.add_fun(Atom::try_from_str("+").unwrap(), 2, |proc, args| {
        assert!(args.len() == 2);
        Ok(erlang::add_2(args[0], args[1], proc).unwrap())
    });

    native
}
