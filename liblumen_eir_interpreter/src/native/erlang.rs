use liblumen_alloc::erts::term::Atom;
use lumen_runtime::otp::erlang;

use crate::module::NativeModule;

pub fn make_erlang() -> NativeModule {
    let mut native = NativeModule::new(Atom::try_from_str("erlang").unwrap());

    native.add_fun(Atom::try_from_str("<").unwrap(), 2, |_proc, args| {
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

    native.add_fun(Atom::try_from_str("monotonic_time").unwrap(), 0, |proc, args| {
        assert!(args.len() == 0);
        Ok(erlang::monotonic_time_0(proc).unwrap())
    });

    native
}
