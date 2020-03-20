use liblumen_alloc::erts::term::prelude::Atom;

use lumen_interpreter::NativeModule;

pub fn make_lumen_web_element() -> NativeModule {
    let mut native = NativeModule::new(Atom::try_from_str("Elixir.Lumen.Web.Element").unwrap());

    native.add_simple(
        Atom::try_from_str("set_attribute").unwrap(),
        3,
        |proc, args| {
            Ok(
                lumen_web::element::set_attribute_3::native(proc, args[0], args[1], args[2])
                    .unwrap(),
            )
        },
    );

    native
}
