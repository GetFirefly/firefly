use std::collections::HashMap;
use std::convert::TryInto;
use std::sync::Arc;

use cranelift_entity::EntityRef;
use libeir_intern::Symbol;
use libeir_ir::constant::{AtomicTerm, Const, ConstKind};
use libeir_ir::{Block, LogicOp, OpKind, PrimOpKind, Value, ValueKind, BinaryEntrySpecifier};

use liblumen_alloc::erts::exception::system;
use liblumen_alloc::erts::process::code::Result;
use liblumen_alloc::erts::process::{ProcessControlBlock, ProcessFlags};
use liblumen_alloc::erts::term::{atom_unchecked, Atom, Term, TypedTerm, AsTerm};
use liblumen_alloc::erts::ModuleFunctionArity;
use liblumen_alloc::erts::process::RootSet;

use crate::module::{ErlangFunction, NativeFunctionKind, ResolvedFunction};
use crate::vm::VMState;

mod r#match;

//macro_rules! trace {
//    ($($t:tt)*) => (lumen_runtime::system::io::puts(&format_args!($($t)*).to_string()))
//}
macro_rules! trace {
    ($($t:tt)*) => ()
}

const VALUE_LIST_MARKER: &str = "eir_value_list_marker_df8gy43h";

pub struct CallExecutor {
    binds: HashMap<Value, Term>,
    next_args: Vec<Term>,
}

pub enum OpResult {
    Block(Block),
    Term(Term),
    TermYield(Term),
}

trait TermCollection {
    unsafe fn add(&mut self, root_set: &mut RootSet);
}
impl TermCollection for &mut CallExecutor {
    unsafe fn add(&mut self, root_set: &mut RootSet) {
        self.binds.add(root_set);
        (&mut self.next_args as &mut [Term]).add(root_set);
    }
}
impl TermCollection for Term {
    unsafe fn add(&mut self, root_set: &mut RootSet) {
        root_set.push(self);
    }
}
impl TermCollection for &mut Term {
    unsafe fn add(&mut self, root_set: &mut RootSet) {
        root_set.push(*self);
    }
}
impl TermCollection for &mut [Term] {
    unsafe fn add(&mut self, root_set: &mut RootSet) {
        for term in self.iter_mut() {
            root_set.push(term);
        }
    }
}
impl<K> TermCollection for HashMap<K, Term> {
    unsafe fn add(&mut self, root_set: &mut RootSet) {
        for term in self.values_mut() {
            root_set.push(term);
        }
    }
}
impl<A, B> TermCollection for (A, B) where A: TermCollection, B: TermCollection {
    unsafe fn add(&mut self, root_set: &mut RootSet) {
        self.0.add(root_set);
        self.1.add(root_set);
    }
}

/// Will keep trying to execute the inner function and performing GC until
/// we succeed without alloc error.
fn try_gc<T, F, R>(proc: &Arc<ProcessControlBlock>, terms: &mut T, fun: &mut F) -> R
where
    T: TermCollection,
    F: FnMut(&mut T) -> std::result::Result<R, system::Exception>,
{
    // Loop, keep trying the inner function until we succeed
    loop {
        match fun(terms) {
            Ok(inner) => break inner,
            Err(system::Exception::Alloc(_)) => {
                let mut heap = proc.acquire_heap();

                let mut rootset = RootSet::new(&mut []);
                // Process dictionary/other process related terms
                proc.base_root_set(&mut rootset);
                // Terms are in root set
                unsafe { terms.add(&mut rootset) };

                trace!("=================================================== GC");
                match heap.garbage_collect(proc, 0, rootset) {
                    Ok(_) => (),
                    Err(_) => {
                        proc.set_flags(ProcessFlags::NeedFullSweep);

                        let mut rootset = RootSet::new(&mut []);
                        // Process dictionary/other process related terms
                        proc.base_root_set(&mut rootset);
                        // Terms are in root set
                        unsafe { terms.add(&mut rootset) };

                        trace!("=================================================== FULLSWEEP GC");
                        match heap.garbage_collect(proc, 0, rootset) {
                            Ok(_) => (),
                            Err(_) => panic!(),
                        }
                    },
                }
            }
        }

    }
}

/// Sets up the current stack frame of `proc` to call `closure` with `args`.
fn call_closure(
    proc: &Arc<ProcessControlBlock>,
    mut closure: Term,
    args: &mut [Term],
) {
    try_gc(proc, &mut (&mut closure, args), &mut |(closure_term, args)| {
        call_closure_inner(proc, **closure_term, closure_term.to_typed_term().unwrap(), args)
    })
}

fn call_closure_inner(
    proc: &Arc<ProcessControlBlock>,
    closure_term: Term,
    closure_typed_term: TypedTerm,
    args: &mut [Term],
) -> Result {
    match closure_typed_term {
        TypedTerm::Closure(closure) => {
            let is_closure = closure.env_len() > 0;

            if is_closure {
                proc.stack_push(closure_term)?;
            }

            let arg_list = proc.list_from_iter(args.iter().cloned())?;
            proc.stack_push(arg_list)?;

            proc.replace_frame(closure.frame());
            Ok(())
        }
        TypedTerm::Boxed(boxed) => {
            call_closure_inner(proc, closure_term, boxed.to_typed_term().unwrap(), args)
        },
        t => panic!("CALL TO: {:?}", t),
    }
}


impl CallExecutor {
    pub fn new() -> Self {
        CallExecutor {
            binds: HashMap::new(),
            next_args: Vec::new(),
        }
    }

    /// Calls the given MFA with args. Will call the entry block.
    pub fn call(
        &mut self,
        vm: &VMState,
        proc: &Arc<ProcessControlBlock>,
        module: Atom,
        function: Atom,
        arity: usize,
        args: &mut [Term],
    ) {
        trace!("======== RUN {} ========", proc.pid());
        let modules = vm.modules.read().unwrap();
        match modules.lookup_function(module, function, arity) {
            None => {
                self.fun_not_found(proc, args[1], module, function, arity).unwrap();
            },
            Some(ResolvedFunction::Native(native)) => {
                assert!(arity == args.len() + 2);
                self.run_native(vm, proc, native, args);
            },
            Some(ResolvedFunction::Erlang(fun)) => {
                let entry = fun.fun.block_entry();
                self.run_erlang(vm, proc, fun, entry, args);
            }
        }
    }

    /// Calls a block in the given MFA with an environment.
    pub fn call_block(
        &mut self,
        vm: &VMState,
        proc: &Arc<ProcessControlBlock>,
        module: Atom,
        function: Atom,
        arity: usize,
        args: &mut [Term],
        block: Block,
        env: &mut [Term],
    ) {
        trace!("======== RUN {} ========", proc.pid());
        let modules = vm.modules.read().unwrap();
        match modules.lookup_function(module, function, arity) {
            None => self.fun_not_found(proc, args[1], module, function, arity).unwrap(),
            Some(ResolvedFunction::Native(_ptr)) => unreachable!(),
            Some(ResolvedFunction::Erlang(fun)) => {
                let live = &fun.live.live[&block];
                assert!(live.size(&fun.live.pool) == env.len());

                for (v, t) in live.iter(&fun.live.pool).zip(env.iter()) {
                    self.binds.insert(v, *t);
                }

                self.run_erlang(vm, proc, fun, block, args);
            }
        }
    }

    fn fun_not_found(&self, _proc: &Arc<ProcessControlBlock>, _throw_cont: Term, module: Atom, function: Atom, arity: usize) -> Result {
        panic!("Undef: {} {} {}", module, function, arity);
        //let exit_atom = atom_unchecked("EXIT");
        //let undef_atom = atom_unchecked("undef");
        //let trace_atom = atom_unchecked("trace");
        //self.call_closure(proc, throw_cont, &[exit_atom, undef_atom, trace_atom])
    }

    fn run_native(
        &mut self,
        _vm: &VMState,
        proc: &Arc<ProcessControlBlock>,
        native: NativeFunctionKind,
        mut args: &mut [Term],
    ) {
        try_gc(proc, &mut args, &mut |args| {
            match native {
                NativeFunctionKind::Simple(ptr) => match ptr(proc, &args[2..]) {
                    Ok(ret) => Ok(call_closure(proc, args[0], &mut [ret])),
                    Err(()) => panic!(),
                },
                NativeFunctionKind::Yielding(ptr) => ptr(proc, args),
            }
        })
    }

    fn run_erlang(
        &mut self,
        vm: &VMState,
        proc: &Arc<ProcessControlBlock>,
        fun: &ErlangFunction,
        mut block: Block,
        args: &mut [Term],
    ) {
        self.next_args.extend(args.iter().cloned());

        let mut exec = self;
        // Outer loop for optimized execution within the current function
        'outer: loop {

            // Insert block argument into environment
            let block_arg_vals = fun.fun.block_args(block);
            trace!("{:?} {:?}", &block_arg_vals, &exec.next_args);
            assert!(block_arg_vals.len() == exec.next_args.len());
            for (v, t) in block_arg_vals.iter().zip(exec.next_args.iter()) {
                exec.binds.insert(*v, t.clone());
            }

            match try_gc(proc, &mut exec, &mut |exec| {
                exec.next_args.clear();
                exec.run_erlang_op(vm, proc, fun, block)
            }) {
                OpResult::Block(b) => {
                    block = b;
                    continue;
                },
                OpResult::Term(t) => break call_closure(proc, t, &mut exec.next_args),
                OpResult::TermYield(t) => break call_closure(proc, t, &mut exec.next_args),
            }
        }
    }

    fn make_const_term(
        &self,
        proc: &Arc<ProcessControlBlock>,
        fun: &ErlangFunction,
        const_val: Const,
    ) -> std::result::Result<Term, system::Exception> {
        let res = match fun.fun.cons().const_kind(const_val) {
            ConstKind::Atomic(AtomicTerm::Atom(atom)) => Ok(atom_unchecked(&atom.0.as_str())),
            ConstKind::Atomic(AtomicTerm::Int(int)) =>
                Ok(proc.integer(int.0)?),
            ConstKind::Atomic(AtomicTerm::Binary(bin)) =>
                Ok(proc.binary_from_bytes(&bin.0)?),
            ConstKind::Tuple { entries } => {
                let vec: std::result::Result<Vec<_>, _> = entries
                    .as_slice(&fun.fun.cons().const_pool)
                    .iter()
                    .map(|e| self.make_const_term(proc, fun, *e))
                    .collect();
                let tup = proc.tuple_from_slice(&vec?)?;
                Ok(tup)
            },
            ConstKind::ListCell { head, tail } => {
                let res = proc.cons(
                    self.make_const_term(proc, fun, *head)?,
                    self.make_const_term(proc, fun, *tail)?,
                )?;
                Ok(res)
            },
            ConstKind::Atomic(AtomicTerm::Nil) => Ok(Term::NIL),
            kind => unimplemented!("{:?}", kind),
        };
        res
    }

    fn make_closure(
        &self,
        proc: &Arc<ProcessControlBlock>,
        fun: &ErlangFunction,
        block: Block,
    ) -> std::result::Result<Term, system::Exception> {
        let live = &fun.live.live[&block];

        // FIXME vec alloc
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

        let closure = proc.closure_with_env_from_slice(
            mfa.into(),
            crate::code::interpreter_closure_code,
            proc.pid_term(),
            &env,
        )?;

        Ok(closure)
    }

    fn make_term(
        &self,
        proc: &Arc<ProcessControlBlock>,
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
                        let mut vec = terms?;
                        vec.insert(0, atom_unchecked(VALUE_LIST_MARKER));
                        Ok(proc.tuple_from_slice(&vec)?)
                    }
                    PrimOpKind::Tuple => {
                        let terms: std::result::Result<Vec<_>, _> = reads.iter()
                            .map(|r| self.make_term(proc, fun, *r))
                            .collect();
                        let vec = terms?;
                        Ok(proc.tuple_from_slice(&vec)?)
                    }
                    PrimOpKind::ListCell => {
                        assert!(reads.len() == 2);
                        let head = self.make_term(proc, fun, reads[0])?;
                        let tail = self.make_term(proc, fun, reads[1])?;
                        let res = proc.cons(head, tail)?;
                        Ok(res)
                    }
                    PrimOpKind::LogicOp(LogicOp::And) => {
                        let mut acc = true;
                        for read in reads.iter() {
                            let term = self.make_term(proc, fun, *read).unwrap();
                            let res: bool = term.try_into().ok().unwrap();
                            acc = acc & res;
                        }
                        Ok(acc.into())
                    }
                    PrimOpKind::LogicOp(LogicOp::Or) => {
                        let mut acc = false;
                        for read in reads.iter() {
                            let term = self.make_term(proc, fun, *read).unwrap();
                            let res: bool = term.try_into().ok().unwrap();
                            acc = acc | res;
                        }
                        Ok(acc.into())
                    }
                    PrimOpKind::CaptureFunction => {
                        let module: Atom = self.make_term(proc, fun, reads[0])?.try_into().unwrap();
                        let function: Atom = self.make_term(proc, fun, reads[1])?.try_into().unwrap();
                        let arity: usize = self.make_term(proc, fun, reads[2])?.try_into().unwrap();

                        let mfa = ModuleFunctionArity {
                            module,
                            function,
                            arity: arity as u8,
                        };

                        Ok(proc.closure_with_env_from_slice(
                            mfa.into(),
                            crate::code::interpreter_mfa_code,
                            proc.pid_term(),
                            &[],
                        )?)
                    }
                    kind => unimplemented!("{:?}", kind),
                }
            }
        }
    }

    fn val_call(
        &mut self,
        proc: &Arc<ProcessControlBlock>,
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
        proc: &Arc<ProcessControlBlock>,
        fun: &ErlangFunction,
        block: Block,
    ) -> std::result::Result<OpResult, system::Exception> {

        let reads = fun.fun.block_reads(block);
        let kind = fun.fun.block_kind(block).unwrap();
        trace!("OP: {:?}", kind);

        proc.reduce();

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
                            match items[0].to_typed_term().unwrap() {
                                TypedTerm::Atom(atom) if atom.name() == VALUE_LIST_MARKER => {
                                    assert!(items.len() - 1 == *num);
                                    for item in items.iter().skip(1) {
                                        self.next_args.push(item);
                                    }
                                    self.val_call(proc, fun, reads[0])
                                }
                                _ => {
                                    self.next_args.push(term);
                                    self.val_call(proc, fun, reads[0])
                                }
                            }
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

                let closure = proc.closure_with_env_from_slice(
                    mfa.into(),
                    crate::code::interpreter_mfa_code,
                    proc.pid_term(),
                    &[],
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
            OpKind::Match { branches } => self::r#match::match_op(self, proc, fun, branches, block),
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
            OpKind::BinaryPush { specifier: BinaryEntrySpecifier::Bytes { .. } } => {
                assert!(reads.len() == 4);
                let head = self.make_term(proc, fun, reads[2])?;
                let tail = self.make_term(proc, fun, reads[3])?;
                trace!("{:?} {:?}", head, tail);

                let mut head_bin: Vec<u8> = head.try_into().unwrap();
                let tail_bin: Vec<u8> = tail.try_into().unwrap();
                head_bin.extend(tail_bin.iter());

                self.next_args.push(proc.binary_from_bytes(&head_bin)?);
                self.val_call(proc, fun, reads[0])
            }
            OpKind::Unreachable => {
                println!("==== Reached OpKind::Unreachable! ====");
                println!("Fun: {} Block: {}", fun.fun.ident(), block);
                unreachable!();
            }
            kind => unimplemented!("{:?}", kind),
        }
    }
}
