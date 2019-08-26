use liblumen_alloc::erts::term::Atom;
//use liblumen_alloc::erts::ModuleFunctionArity;
//use liblumen_alloc::erts::process::code::stack::frame::Placement;
use lumen_runtime::otp::lists;

use crate::module::NativeModule;

pub fn make_lists() -> NativeModule {
    let mut native = NativeModule::new(Atom::try_from_str("lists").unwrap());

    native.add_simple(Atom::try_from_str("keyfind").unwrap(), 3, |_proc, args| {
        let res = lists::keyfind_3::native(args[0], args[1], args[2]).unwrap();
        Ok(res)
    });

    native.add_simple(Atom::try_from_str("member").unwrap(), 2, |_proc, args| {
        let res = lists::member_2::native(args[0], args[1]).unwrap();
        Ok(res)
    });

    native
}
