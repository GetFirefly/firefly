use liblumen_alloc::erts::term::Atom;

use liblumen_eir_interpreter::NativeModule;

pub fn make_lumen_web_window() -> NativeModule {
    let mut native = NativeModule::new(Atom::try_from_str("Elixir.Lumen.Web.Window").unwrap());

    native.add_simple(Atom::try_from_str("window").unwrap(), 0, |proc, args| {
        assert!(args.len() == 0);
        Ok(lumen_web::window::window_0::native(proc).unwrap())
    });

    native.add_simple(Atom::try_from_str("document").unwrap(), 1, |proc, args| {
        assert!(args.len() == 1);
        Ok(lumen_web::window::document_1::native(proc, args[0]).unwrap())
    });

    native
}
