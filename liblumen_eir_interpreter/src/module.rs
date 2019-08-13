use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use crate::VMState;

use libeir_intern::Symbol;
use libeir_ir::{Function, FunctionIdent, LiveValues, Module};

use liblumen_alloc::erts::process::code::Result;
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::{Atom, Term};

pub enum ResolvedFunction<'a> {
    Native(NativeFunctionKind),
    Erlang(&'a ErlangFunction),
}

pub struct ModuleRegistry {
    map: HashMap<Atom, ModuleType>,
}

impl ModuleRegistry {
    pub fn new() -> Self {
        ModuleRegistry {
            map: HashMap::new(),
        }
    }

    pub fn register_erlang_module(&mut self, module: Module) {
        let erl_module = ErlangModule::from_eir(module);
        match self.map.remove(&erl_module.name) {
            None => self
                .map
                .insert(erl_module.name, ModuleType::Erlang(erl_module)),
            Some(ModuleType::Native(native)) => self
                .map
                .insert(erl_module.name, ModuleType::Overlayed(erl_module, native)),
            _ => panic!(),
        };
    }

    pub fn register_native_module(&mut self, native: NativeModule) {
        match self.map.remove(&native.name) {
            None => self.map.insert(native.name, ModuleType::Native(native)),
            Some(ModuleType::Erlang(erl)) => self
                .map
                .insert(native.name, ModuleType::Overlayed(erl, native)),
            _ => panic!(),
        };
    }

    pub fn lookup_function(
        &self,
        module: Atom,
        function: Atom,
        arity: usize,
    ) -> Option<ResolvedFunction> {
        println!("LOOKUP {:?}:{:?}/{}", module, function, arity,);
        match self.map.get(&module) {
            None => None,
            Some(ModuleType::Erlang(erl)) => erl
                .functions
                .get(&(function, arity))
                .map(ResolvedFunction::Erlang),
            Some(ModuleType::Native(nat)) => nat
                .functions
                .get(&(function, arity))
                .cloned()
                .map(ResolvedFunction::Native),
            Some(ModuleType::Overlayed(erl, nat)) => {
                if let Some(nat_fun) = nat.functions.get(&(function, arity)) {
                    Some(ResolvedFunction::Native(*nat_fun))
                } else {
                    erl.functions
                        .get(&(function, arity))
                        .map(ResolvedFunction::Erlang)
                }
            }
        }
    }
}

pub enum NativeReturn {
    Return { term: Rc<Term> },
    Throw { typ: Rc<Term>, reason: Rc<Term> },
}

#[derive(Copy, Clone)]
pub enum NativeFunctionKind {
    Simple(fn(&Arc<ProcessControlBlock>, &[Term]) -> std::result::Result<Term, ()>),
    Yielding(fn(&Arc<ProcessControlBlock>, &[Term]) -> Result),
}

pub struct NativeModule {
    pub name: Atom,
    pub functions: HashMap<(Atom, usize), NativeFunctionKind>,
}
impl NativeModule {
    pub fn new(name: Atom) -> Self {
        NativeModule {
            name,
            functions: HashMap::new(),
        }
    }

    pub fn add_simple(
        &mut self,
        name: Atom,
        arity: usize,
        fun: fn(&Arc<ProcessControlBlock>, &[Term]) -> std::result::Result<Term, ()>,
    ) {
        self.functions
            .insert((name, arity), NativeFunctionKind::Simple(fun));
    }

    pub fn add_yielding(
        &mut self,
        name: Atom,
        arity: usize,
        fun: fn(&Arc<ProcessControlBlock>, &[Term]) -> Result,
    ) {
        self.functions
            .insert((name, arity), NativeFunctionKind::Yielding(fun));
    }
}

pub struct ErlangFunction {
    pub fun: Function,
    pub live: LiveValues,
}

pub struct ErlangModule {
    pub name: Atom,
    pub functions: HashMap<(Atom, usize), ErlangFunction>,
}

impl ErlangModule {
    pub fn from_eir(module: Module) -> Self {
        let name_atom = Atom::try_from_str(module.name.as_str()).unwrap();
        let functions = module
            .functions
            .values()
            .map(|fun| {
                let nfun = ErlangFunction {
                    live: fun.live_values(),
                    fun: fun.clone(),
                };
                let name = Atom::try_from_str(fun.ident().name.as_str()).unwrap();
                ((name, fun.ident().arity), nfun)
            })
            .collect();
        ErlangModule {
            name: name_atom,
            functions,
        }
    }
}

pub enum ModuleType {
    Erlang(ErlangModule),
    Overlayed(ErlangModule, NativeModule),
    Native(NativeModule),
}
