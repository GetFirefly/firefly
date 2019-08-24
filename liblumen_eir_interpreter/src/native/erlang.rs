use liblumen_alloc::erts::term::{atom_unchecked, Atom, Term, TypedTerm};
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
    native.add_simple(Atom::try_from_str(">=").unwrap(), 2, |_proc, args| {
        assert!(args.len() == 2);
        Ok(erlang::is_greater_than_or_equal_2(args[0], args[1]))
    });
    native.add_simple(Atom::try_from_str("==").unwrap(), 2, |_proc, args| {
        assert!(args.len() == 2);
        Ok(erlang::are_equal_after_conversion_2(args[0], args[1]))
    });
    native.add_simple(Atom::try_from_str("=:=").unwrap(), 2, |_proc, args| {
        assert!(args.len() == 2);
        Ok(erlang::are_exactly_equal_2(args[0], args[1]))
    });

    native.add_simple(Atom::try_from_str("spawn_opt").unwrap(), 4, |proc, args| {
        assert!(args.len() == 4);

        match args[3].to_typed_term().unwrap() {
            TypedTerm::List(cons) => {
                let mut iter = cons.into_iter();
                assert!(iter.next() == Some(Ok(atom_unchecked("link").into())));
                assert!(iter.next() == None);
            }
            t => panic!("{:?}", t),
        }

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
            ).unwrap()
        };

        println!("SPAWN M: {:?}", args[0]);
        println!("SPAWN F: {:?}", args[1]);
        println!("SPAWN A: {:?}", args[2]);
        println!("SPAWN OPTS: {:?}", args[3]);

        //match args[2].to_typed_term().unwrap() {
        //    TypedTerm::List(cons) => println!("LENBEFORE {}", cons.into_iter().count()),
        //    _ => panic!(),
        //}

        let inner_args = proc.cons(ret, proc.cons(ret, args[2]).unwrap()).unwrap();
        //match inner_args.to_typed_term().unwrap() {
        //    TypedTerm::List(cons) => println!("LENAFTER {}", cons.into_iter().count()),
        //    _ => panic!(),
        //}

        //match inner_args.to_typed_term().unwrap() {
        //    TypedTerm::List(cons) => {
        //        for v in cons.into_iter() {
        //            println!("SPAWN SINGLEARG {:?}", v);
        //        }
        //    },
        //    _ => panic!(),
        //}

        let res = erlang::spawn_link_3::native(proc, args[0], args[1], inner_args).unwrap();
        Ok(res)
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

    native.add_simple(Atom::try_from_str("monitor").unwrap(), 2, |_proc, _args| {
        Ok(Term::NIL)
    });
    native.add_simple(Atom::try_from_str("demonitor").unwrap(), 2, |_proc, _args| {
        Ok(Term::NIL)
    });

    native.add_simple(Atom::try_from_str("send").unwrap(), 2, |proc, args| {
        assert!(args.len() == 2);
        Ok(erlang::send_2(args[0], args[1], proc).unwrap())
    });
    native.add_simple(Atom::try_from_str("send").unwrap(), 3, |proc, args| {
        assert!(args.len() == 3);
        //assert!(args[2] == Term::NIL);
        Ok(erlang::send_2(args[0], args[1], proc).unwrap())
    });
    native.add_simple(Atom::try_from_str("!").unwrap(), 2, |proc, args| {
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
    native.add_simple(Atom::try_from_str("is_list").unwrap(), 1, |_proc, args| {
        assert!(args.len() == 1);
        Ok(erlang::is_list_1(args[0]))
    });
    native.add_simple(Atom::try_from_str("is_atom").unwrap(), 1, |_proc, args| {
        assert!(args.len() == 1);
        Ok(erlang::is_atom_1(args[0]))
    });
    native.add_simple(Atom::try_from_str("is_pid").unwrap(), 1, |_proc, args| {
        assert!(args.len() == 1);
        Ok(erlang::is_pid_1(args[0]))
    });
    native.add_simple(Atom::try_from_str("is_function").unwrap(), 1, |_proc, args| {
        assert!(args.len() == 1);
        println!("ISFUN {:?}", args);
        Ok(erlang::is_function_1(args[0]))
    });

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

    native.add_simple(Atom::try_from_str("node").unwrap(), 0, |_proc, args| {
        assert!(args.len() == 0);
        Ok(erlang::node_0())
    });
    native.add_simple(Atom::try_from_str("node").unwrap(), 1, |_proc, args| {
        assert!(args.len() == 1);
        Ok(atom_unchecked("nonode@nohost"))
    });
    native.add_simple(Atom::try_from_str("whereis").unwrap(), 1, |_proc, args| {
        assert!(args.len() == 1);
        Ok(erlang::whereis_1(args[0]).unwrap())
    });

    native.add_simple(Atom::try_from_str("process_info").unwrap(), 2, |_proc, args| {
        assert!(args.len() == 2);
        match args[1].to_typed_term().unwrap() {
            TypedTerm::Atom(atom) if atom.name() == "registered_name" => {
                Ok(args[0])
            }
            _ => panic!()
        }
    });

    native.add_simple(Atom::try_from_str("get").unwrap(), 1, |proc, args| {
        assert!(args.len() == 1);
        Ok(proc.get(args[0]))
    });
    native.add_simple(Atom::try_from_str("put").unwrap(), 2, |proc, args| {
        assert!(args.len() == 2);
        Ok(proc.put(args[0], args[1]).unwrap())
    });

    native
}
