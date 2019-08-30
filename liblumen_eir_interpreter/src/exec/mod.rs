use std::collections::HashMap;
use std::convert::TryInto;
use std::sync::Arc;

use cranelift_entity::EntityRef;
use libeir_intern::Symbol;
use libeir_ir::constant::{AtomicTerm, Const, ConstKind};
use libeir_ir::{Block, OpKind, PrimOpKind, Value, ValueKind};

use liblumen_alloc::erts::exception::system;
use liblumen_alloc::erts::process::code::Result;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{atom_unchecked, Atom, Term, TypedTerm};
use liblumen_alloc::erts::ModuleFunctionArity;

use crate::module::{ErlangFunction, NativeFunctionKind, ResolvedFunction};
use crate::vm::VMState;

mod r#match;

pub struct CallExecutor {
    binds: HashMap<Value, Term>,
    next_args: Vec<Term>,
}

pub enum OpResult {
    Block(Block),
    Term(Term),
    TermYield(Term),
}

impl CallExecutor {
    pub fn new() -> Self {
        CallExecutor {
            binds: HashMap::new(),
            next_args: Vec::new(),
        }
    }

    pub fn call(
        &mut self,
        vm: &VMState,
        proc: &Arc<Process>,
        module: Atom,
        function: Atom,
        arity: usize,
        args: &[Term],
    ) -> Result {
        let modules = vm.modules.read().unwrap();
        match modules.lookup_function(module, function, arity) {
            None => self.fun_not_found(proc, args[1]),
            Some(ResolvedFunction::Native(native)) => self.run_native(vm, proc, native, args),
            Some(ResolvedFunction::Erlang(fun)) => {
                let entry = fun.fun.block_entry();
                self.run_erlang(vm, proc, fun, entry, args)
            }
        }
    }

    pub fn call_block(
        &mut self,
        vm: &VMState,
        proc: &Arc<Process>,
        module: Atom,
        function: Atom,
        arity: usize,
        args: &[Term],
        block: Block,
        env: &[Term],
    ) -> Result {
        let modules = vm.modules.read().unwrap();
        println!("ARGS {:?}", args);
        match modules.lookup_function(module, function, arity) {
            None => self.fun_not_found(proc, args[1]),
            Some(ResolvedFunction::Native(_ptr)) => unreachable!(),
            Some(ResolvedFunction::Erlang(fun)) => {
                let live = &fun.live.live[&block];
                assert!(live.size(&fun.live.pool) == env.len());

                for (v, t) in live.iter(&fun.live.pool).zip(env.iter()) {
                    self.binds.insert(v, *t);
                }

                self.run_erlang(vm, proc, fun, block, args)
            }
        }
    }

    fn fun_not_found(&self, proc: &Arc<Process>, throw_cont: Term) -> Result {
        let exit_atom = atom_unchecked("EXIT");
        let undef_atom = atom_unchecked("undef");
        let trace_atom = atom_unchecked("trace");
        self.call_closure(proc, throw_cont, &[exit_atom, undef_atom, trace_atom])
    }

    fn call_closure(&self, proc: &Arc<Process>, closure: Term, args: &[Term]) -> Result {
        match closure.to_typed_term().unwrap() {
            TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                TypedTerm::Closure(closure) => {
                    //assert!(closure.env_hack.len() != 1);
                    if closure.env.len() > 0 {
                        let env_list = proc.list_from_slice(&closure.env[1..]).unwrap();
                        proc.stack_push(env_list)?;

                        let block_id = closure.env[0];
                        proc.stack_push(block_id)?;
                    }

                    let arg_list = proc.list_from_iter(args.iter().cloned()).unwrap();
                    proc.stack_push(arg_list)?;

                    if closure.env.len() > 0 {
                        let mfa = closure.module_function_arity();
                        proc.stack_push(proc.integer(mfa.arity).unwrap()).unwrap();
                    }

                    proc.replace_frame(closure.frame());
                    //Process::call_code(proc)
                    Ok(())
                }
                _ => panic!(),
            },
            _ => panic!(),
        }
    }

    fn run_native(
        &mut self,
        _vm: &VMState,
        proc: &Arc<Process>,
        native: NativeFunctionKind,
        args: &[Term],
    ) -> Result {
        match native {
            NativeFunctionKind::Simple(ptr) => match ptr(proc, &args[2..]) {
                Ok(ret) => self.call_closure(proc, args[0], &[ret]),
                Err(()) => panic!(),
            },
            NativeFunctionKind::Yielding(ptr) => ptr(proc, args),
        }
    }

    fn run_erlang(
        &mut self,
        vm: &VMState,
        proc: &Arc<Process>,
        fun: &ErlangFunction,
        mut block: Block,
        args: &[Term],
    ) -> Result {
        self.next_args.extend(args.iter().cloned());

        loop {
            let block_arg_vals = fun.fun.block_args(block);
            assert!(block_arg_vals.len() == self.next_args.len());
            for (v, t) in block_arg_vals.iter().zip(self.next_args.iter()) {
                self.binds.insert(*v, t.clone());
            }
            self.next_args.clear();

            match self.run_erlang_op(vm, proc, fun, block).unwrap() {
                OpResult::Block(b) => block = b,
                OpResult::Term(t) => break self.call_closure(proc, t, &self.next_args),
                OpResult::TermYield(t) => break self.call_closure(proc, t, &self.next_args),
            }
        }
    }

    fn make_const_term(
        &self,
        proc: &Arc<Process>,
        fun: &ErlangFunction,
        const_val: Const,
    ) -> std::result::Result<Term, system::Exception> {
        match fun.fun.cons().const_kind(const_val) {
            ConstKind::Atomic(AtomicTerm::Atom(atom)) => Ok(atom_unchecked(&atom.0.as_str())),
            ConstKind::Atomic(AtomicTerm::Int(int)) => Ok(proc.integer(int.0)?),
            //ConstKind::Atomic(AtomicTerm::BigInt(int)) => {
            //    Term::Integer(int.0.clone()).into()
            //}
            //ConstKind::Atomic(AtomicTerm::Float(flt)) => {
            //    Term::Float(flt.0.into()).into()
            //}
            //ConstKind::Atomic(AtomicTerm::Binary(bin)) => {
            //    Term::Binary(Rc::new(bin.0.clone().into())).into()
            //}
            ConstKind::Atomic(AtomicTerm::Nil) => Ok(Term::NIL),
            //ConstKind::ListCell { head, tail } => {
            //    Term::ListCell(
            //        self.make_const_term(fun, *head),
            //        self.make_const_term(fun, *tail),
            //    ).into()
            //}
            //ConstKind::Tuple { entries } => {
            //    let vec = entries.as_slice(&fun.fun.cons().const_pool)
            //        .iter()
            //        .map(|e| self.make_const_term(fun, *e))
            //        .collect::<Vec<_>>();
            //    Term::Tuple(vec).into()
            //}
            //ConstKind::Map { keys, values } => {
            //    assert!(keys.len(&fun.fun.cons().const_pool)
            //            == values.len(&fun.fun.cons().const_pool));

            //    let mut map = MapTerm::new();
            //    for (key, val) in keys.as_slice(&fun.fun.cons().const_pool).iter()
            //        .zip(values.as_slice(&fun.fun.cons().const_pool).iter())
            //    {
            //        let key_v = self.make_const_term(fun, *key);
            //        let val_v = self.make_const_term(fun, *val);
            //        map.insert(key_v, val_v);
            //    }

            //    Term::Map(map).into()
            //}
            kind => unimplemented!("{:?}", kind),
        }
    }

    fn make_closure(
        &self,
        proc: &Arc<Process>,
        fun: &ErlangFunction,
        block: Block,
    ) -> std::result::Result<Term, system::Exception> {
        let live = &fun.live.live[&block];

        let mut env = Vec::new();
        env.push(proc.integer(block.index())?);
        for v in live.iter(&fun.live.pool) {
            assert!(fun.fun.value_argument(v).is_some());
            env.push(self.make_term(proc, fun, v)?);
        }

        let mfa = ModuleFunctionArity {
            module: Atom::try_from_str(fun.fun.ident().module.as_str()).unwrap(),
            function: Atom::try_from_str(fun.fun.ident().name.as_str()).unwrap(),
            arity: fun.fun.ident().arity as u8,
        };

        let closure = proc.closure(
            proc.pid_term(),
            mfa.into(),
            crate::code::interpreter_closure_code,
            env,
        )?;

        Ok(closure)
    }

    fn make_term(
        &self,
        proc: &Arc<Process>,
        fun: &ErlangFunction,
        value: Value,
    ) -> std::result::Result<Term, system::Exception> {
        match fun.fun.value_kind(value) {
            ValueKind::Block(block) => self.make_closure(proc, fun, block),
            ValueKind::Argument(_, _) => Ok(self.binds[&value]),
            ValueKind::Const(cons) => self.make_const_term(proc, fun, cons),
            ValueKind::PrimOp(prim) => {
                let reads = fun.fun.primop_reads(prim);
                match fun.fun.primop_kind(prim) {
                    PrimOpKind::ValueList => {
                        let terms: std::result::Result<Vec<_>, _> = reads
                            .iter()
                            .map(|r| self.make_term(proc, fun, *r))
                            .collect();
                        Ok(proc.tuple_from_slice(&terms?)?)
                    }
                    //PrimOpKind::Tuple => {
                    //    let terms: Vec<_> = reads.iter()
                    //        .map(|r| self.make_term(fun, *r)).collect();
                    //    Term::Tuple(terms).into()
                    //}
                    PrimOpKind::ListCell => {
                        assert!(reads.len() == 2);
                        let head = self.make_term(proc, fun, reads[0])?;
                        let tail = self.make_term(proc, fun, reads[1])?;
                        Ok(proc.cons(head, tail)?)
                    }
                    //PrimOpKind::BinOp(BinOp::Equal) => {
                    //    assert!(reads.len() == 2);
                    //    let lhs = self.make_term(fun, reads[0]);
                    //    let rhs = self.make_term(fun, reads[1]);
                    //    Term::new_bool(lhs.erl_eq(&*rhs)).into()
                    //}
                    kind => unimplemented!("{:?}", kind),
                }
            }
        }
    }

    fn val_call(
        &mut self,
        proc: &Arc<Process>,
        fun: &ErlangFunction,
        value: Value,
    ) -> std::result::Result<OpResult, system::Exception> {
        if let ValueKind::Block(block) = fun.fun.value_kind(value) {
            Ok(OpResult::Block(block))
        } else {
            let term = self.make_term(proc, fun, value)?;
            Ok(OpResult::Term(term))
        }
    }

    fn run_erlang_op(
        &mut self,
        _vm: &VMState,
        proc: &Arc<Process>,
        fun: &ErlangFunction,
        block: Block,
    ) -> std::result::Result<OpResult, system::Exception> {
        let reads = fun.fun.block_reads(block);
        let kind = fun.fun.block_kind(block).unwrap();
        println!("OP: {:?}", kind);
        match kind {
            OpKind::Call => {
                for read in reads.iter().skip(1) {
                    let term = self.make_term(proc, fun, *read)?;
                    self.next_args.push(term);
                }
                self.val_call(proc, fun, reads[0])
            }
            OpKind::UnpackValueList(num) => {
                assert!(reads.len() == 2);
                let term = self.make_term(proc, fun, reads[1])?;
                match term.to_typed_term().unwrap() {
                    TypedTerm::Boxed(inner) => match inner.to_typed_term().unwrap() {
                        TypedTerm::Tuple(items) => {
                            assert!(items.len() == *num);
                            for item in items.iter() {
                                self.next_args.push(item);
                            }
                            self.val_call(proc, fun, reads[0])
                        }
                        _ => {
                            self.next_args.push(term);
                            self.val_call(proc, fun, reads[0])
                        }
                    },
                    _ => {
                        self.next_args.push(term);
                        self.val_call(proc, fun, reads[0])
                    }
                }
            }
            OpKind::CaptureFunction => {
                let module: Atom = self.make_term(proc, fun, reads[1])?.try_into().unwrap();
                let function: Atom = self.make_term(proc, fun, reads[2])?.try_into().unwrap();
                let arity: usize = self.make_term(proc, fun, reads[3])?.try_into().unwrap();

                let mfa = ModuleFunctionArity {
                    module,
                    function,
                    arity: arity as u8,
                };

                let closure = proc.closure(
                    proc.pid_term(),
                    mfa.into(),
                    crate::code::interpreter_mfa_code,
                    vec![],
                )?;

                self.next_args.push(closure);
                self.val_call(proc, fun, reads[0])
            }
            OpKind::Intrinsic(name) if *name == Symbol::intern("bool_and") => {
                let mut res = true;
                for val in reads[1..].iter() {
                    let term = self.make_term(proc, fun, *val)?;
                    let b: bool = term.try_into().ok().unwrap();
                    res = res & b;
                }

                self.next_args.push(res.into());
                self.val_call(proc, fun, reads[0])
            }
            OpKind::Intrinsic(name) if *name == Symbol::intern("bool_or") => {
                let mut res = false;
                for val in reads[1..].iter() {
                    let term = self.make_term(proc, fun, *val)?;
                    let b: bool = term.try_into().ok().unwrap();
                    res = res | b;
                }

                self.next_args.push(res.into());
                self.val_call(proc, fun, reads[0])
            }
            OpKind::IfBool => {
                let call_n = if reads.len() == 4 {
                    let bool_term = self.make_term(proc, fun, reads[3]).unwrap();
                    let b: std::result::Result<bool, _> = bool_term.try_into();
                    match b {
                        Ok(true) => 0,
                        Ok(false) => 1,
                        Err(_) => 2,
                    }
                } else if reads.len() == 3 {
                    let bool_term = self.make_term(proc, fun, reads[2]).unwrap();
                    let b: std::result::Result<bool, _> = bool_term.try_into();
                    match b {
                        Ok(true) => 0,
                        Ok(false) => 1,
                        Err(_) => unreachable!(),
                    }
                } else {
                    unreachable!()
                };

                self.val_call(proc, fun, reads[call_n])
            }
            //OpKind::TraceCaptureRaw => {
            //    TermCall {
            //        fun: self.make_term(fun, reads[0]),
            //        args: vec![Term::Nil.into()],
            //    }
            //}
            OpKind::Match { branches } => self::r#match::match_op(self, proc, fun, branches, block),
            //OpKind::BinaryPush { specifier } => {
            //    let bin_term = self.make_term(fun, reads[2]);
            //    let mut bin = match &*bin_term {
            //        Term::Binary(bin) => (**bin).clone(),
            //        Term::BinarySlice { buf, bit_offset, bit_length } => {
            //            let slice = BitSlice::with_offset_length(
            //                &**buf, *bit_offset, *bit_length);
            //            let mut new = BitVec::new();
            //            new.push(slice);
            //            new
            //        }
            //        _ => panic!(),
            //    };

            //    let val_term = self.make_term(fun, reads[3]);

            //    assert!(reads.len() == 4 || reads.len() == 5);
            //    let size_term = reads.get(4).map(|r| self.make_term(fun, *r));

            //    match specifier {
            //        BinaryEntrySpecifier::Integer {
            //            signed, unit, endianness } =>
            //        {
            //            let size = size_term.unwrap().as_usize().unwrap();
            //            let bit_size = *unit as usize * size;

            //            let endian = match *endianness {
            //                Endianness::Big => Endian::Big,
            //                Endianness::Little => Endian::Little,
            //                Endianness::Native => Endian::Big,
            //            };

            //            let val = val_term.as_integer().unwrap().clone();
            //            let carrier = integer_to_carrier(
            //                val, bit_size, endian);

            //            bin.push(carrier);
            //        }
            //        BinaryEntrySpecifier::Bytes { unit: 1 } => {
            //            let binary = val_term.as_binary().unwrap();

            //            if let Some(size_term) = size_term {
            //                dbg!(&size_term, &binary);
            //                assert!(size_term.as_usize().unwrap() == binary.len());
            //            }

            //            bin.push(binary);
            //        }
            //        k => unimplemented!("{:?}", k),
            //    }

            //    return TermCall {
            //        fun: self.make_term(fun, reads[0]),
            //        args: vec![Term::Binary(bin.into()).into()],
            //    };
            //}
            OpKind::MapPut { action } => {
                let map_read = reads[2];
                if let Some(constant) = fun.fun.value_const(map_read) {
                    if let ConstKind::Map { keys, .. } = fun.fun.cons().const_kind(constant) {
                        if keys.len(&fun.fun.cons().const_pool) == 0 {
                            let mut vec = Vec::new();

                            let mut idx = 3;
                            for _ in action.iter() {
                                let key = self.make_term(proc, fun, reads[idx])?;
                                let val = self.make_term(proc, fun, reads[idx + 1])?;
                                idx += 2;

                                vec.push((key, val));
                            }

                            self.next_args.push(proc.map_from_slice(&vec)?);
                            return self.val_call(proc, fun, reads[0]);
                        }
                    }
                }

                unimplemented!()
            }
            OpKind::Intrinsic(name) if *name == Symbol::intern("receive_start") => {
                assert!(reads.len() == 2);

                let timeout = self.make_term(proc, fun, reads[1])?;
                // Only infinity supported
                assert!(timeout == atom_unchecked("infinity"));

                proc.mailbox.lock().borrow_mut().recv_start();

                self.next_args.push(Term::NIL);
                self.val_call(proc, fun, reads[0])
            }
            OpKind::Intrinsic(name) if *name == Symbol::intern("receive_wait") => {
                assert!(reads.len() == 2);

                let mailbox_lock = proc.mailbox.lock();
                let mut mailbox = mailbox_lock.borrow_mut();
                if let Some(msg_term) = mailbox.recv_peek() {
                    mailbox.recv_increment();

                    std::mem::drop(mailbox);
                    std::mem::drop(mailbox_lock);

                    self.next_args.push(msg_term);
                    self.val_call(proc, fun, reads[1])
                } else {
                    // If there are no messages, schedule a call
                    // to the current block for later.
                    let curr_cont = self.make_closure(proc, fun, block).unwrap();
                    self.next_args.push(Term::NIL);
                    proc.wait();
                    Ok(OpResult::TermYield(curr_cont))
                }
            }
            OpKind::Intrinsic(name) if *name == Symbol::intern("receive_done") => {
                assert!(reads.len() >= 1);

                let mailbox_lock = proc.mailbox.lock();
                let mut mailbox = mailbox_lock.borrow_mut();

                if mailbox.recv_last_off_heap() {
                    // Copy to process heap
                    unimplemented!()
                } else {
                    for n in 0..(reads.len() - 1) {
                        let term = self.make_term(proc, fun, reads[n + 1]).unwrap();
                        self.next_args.push(term);
                    }
                }

                mailbox.recv_finish(proc);

                self.val_call(proc, fun, reads[0])
            }
            //OpKind::Unreachable => {
            //    println!("==== Reached OpKind::Unreachable! ====");
            //    println!("Fun: {} Block: {}", fun.fun.ident(), block);
            //    unreachable!();
            //}
            kind => unimplemented!("{:?}", kind),
        }
    }
}
