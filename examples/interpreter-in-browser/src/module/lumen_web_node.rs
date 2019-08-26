use liblumen_alloc::erts::term::{atom_unchecked, Atom};

use liblumen_eir_interpreter::NativeModule;

macro_rules! trace {
    ($($t:tt)*) => (lumen_runtime::system::io::puts(&format_args!($($t)*).to_string()))
}

pub fn make_lumen_web_node() -> NativeModule {
    let mut native = NativeModule::new(Atom::try_from_str("Elixir.Lumen.Web.Node").unwrap());

    native.add_simple(Atom::try_from_str("append_child").unwrap(), 2, |_proc, args| {
        assert!(args.len() == 2);
        Ok(lumen_web::node::append_child_2::native(args[0], args[1]).unwrap())
    });

    native
}
