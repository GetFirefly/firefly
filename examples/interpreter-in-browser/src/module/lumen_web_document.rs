use liblumen_alloc::erts::term::{atom_unchecked, Atom};

use liblumen_eir_interpreter::NativeModule;

macro_rules! trace {
    ($($t:tt)*) => (lumen_runtime::system::io::puts(&format_args!($($t)*).to_string()))
}

pub fn make_lumen_web_document() -> NativeModule {
    let mut native = NativeModule::new(Atom::try_from_str("Elixir.Lumen.Web.Document").unwrap());

    native.add_simple(Atom::try_from_str("body").unwrap(), 1, |proc, args| {
        Ok(lumen_web::document::body_1::native(proc, args[0]).unwrap())
    });

    native.add_simple(
        Atom::try_from_str("create_element").unwrap(),
        2,
        |proc, args| {
            Ok(lumen_web::document::create_element_2::native(proc, args[0], args[1]).unwrap())
        },
    );

    native.add_simple(
        Atom::try_from_str("create_text_node").unwrap(),
        2,
        |proc, args| {
            Ok(lumen_web::document::create_text_node_2::native(proc, args[0], args[1]).unwrap())
        },
    );

    native.add_simple(
        Atom::try_from_str("get_element_by_id").unwrap(),
        2,
        |proc, args| {
            Ok(lumen_web::document::get_element_by_id_2::native(proc, args[0], args[1]).unwrap())
        },
    );

    native
}
