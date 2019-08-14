use liblumen_alloc::erts::term::Atom;
use liblumen_alloc::erts::ModuleFunctionArity;
use lumen_runtime::otp::erlang;

use crate::module::NativeModule;

pub fn make_erlang() -> NativeModule {
    let mut native = NativeModule::new(Atom::try_from_str("erlang").unwrap());

    native.add_simple(Atom::try_from_str("<").unwrap(), 2, |_proc, args| {
        assert!(args.len() == 2);
        Ok(erlang::is_less_than_2(args[0], args[1]))
    });
    native.add_simple(Atom::try_from_str("=<").unwrap(), 2, |_proc, args| {
        assert!(args.len() == 2);
        Ok(erlang::is_equal_or_less_than_2(args[0], args[1]))
    });
    native.add_simple(Atom::try_from_str("=:=").unwrap(), 2, |_proc, args| {
        assert!(args.len() == 2);
        Ok(erlang::are_exactly_equal_2(args[0], args[1]))
    });

    native.add_simple(Atom::try_from_str("spawn").unwrap(), 3, |proc, args| {
        assert!(args.len() == 3);

        let ret = {
            let mfa = ModuleFunctionArity {
                module: Atom::try_from_str("lumen_eir_interpreter_intrinsics").unwrap(),
                function: Atom::try_from_str("return_clean").unwrap(),
                arity: 1,
            };
            proc.closure(
                proc.pid_term(),
                mfa.into(),
                crate::code::return_clean,
                vec![],
            )
            .unwrap()
        };

        let inner_args = proc.cons(ret, proc.cons(ret, args[2]).unwrap()).unwrap();
        Ok(erlang::spawn_3::native(proc, args[0], args[1], inner_args).unwrap())
    });

    native.add_simple(Atom::try_from_str("send").unwrap(), 2, |proc, args| {
        assert!(args.len() == 2);
        Ok(erlang::send_2(args[0], args[1], proc).unwrap())
    });

    native.add_simple(Atom::try_from_str("-").unwrap(), 2, |proc, args| {
        assert!(args.len() == 2);
        Ok(erlang::subtract_2::native(proc, args[0], args[1]).unwrap())
    });

    native.add_simple(Atom::try_from_str("+").unwrap(), 2, |proc, args| {
        assert!(args.len() == 2);
        Ok(erlang::add_2::native(proc, args[0], args[1]).unwrap())
    });

    native.add_simple(Atom::try_from_str("self").unwrap(), 0, |proc, args| {
        assert!(args.len() == 0);
        Ok(proc.pid_term())
    });

    native.add_simple(
        Atom::try_from_str("is_integer").unwrap(),
        1,
        |_proc, args| {
            assert!(args.len() == 1);
            Ok(erlang::is_integer_1(args[0]))
        },
    );

    native.add_simple(
        Atom::try_from_str("monotonic_time").unwrap(),
        0,
        |proc, args| {
            assert!(args.len() == 0);
            Ok(erlang::monotonic_time_0::native(proc).unwrap())
        },
    );

    native.add_yielding(Atom::try_from_str("apply").unwrap(), 3, |proc, args| {
        assert!(args.len() == 5);

        let inner_args = proc.cons(args[0], proc.cons(args[1], args[4])?)?;
        proc.stack_push(inner_args)?;

        proc.stack_push(args[3])?;
        proc.stack_push(args[2])?;

        crate::code::apply(proc)
    });

    native
}
