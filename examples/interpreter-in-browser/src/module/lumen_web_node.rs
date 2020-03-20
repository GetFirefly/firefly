use liblumen_alloc::erts::term::prelude::Atom;

use lumen_interpreter::NativeModule;

pub fn make_lumen_web_node() -> NativeModule {
    let mut native = NativeModule::new(Atom::try_from_str("Elixir.Lumen.Web.Node").unwrap());

    native.add_simple(
        Atom::try_from_str("append_child").unwrap(),
        2,
        |proc, args| Ok(lumen_web::node::append_child_2::native(proc, args[0], args[1]).unwrap()),
    );

    native
}
