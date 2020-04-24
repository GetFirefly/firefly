use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use libeir_ir::{Function, FunctionIndex, LiveValues, Module};

use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::erts::process::frames::Result;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

macro_rules! trace {
    ($($t:tt)*) => (crate::runtime::sys::io::puts(&format_args!($($t)*).to_string()))
}
//macro_rules! trace {
//    ($($t:tt)*) => ()
//}

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
        trace!("LOOKUP {}:{}/{}", module, function, arity);
        match self.map.get(&module) {
            None => None,
            Some(ModuleType::Erlang(erl)) => erl
                .name_map
                .get(&(function, arity))
                .map(|i| &erl.funs[i])
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
                    erl.name_map
                        .get(&(function, arity))
                        .map(|i| &erl.funs[i])
                        .map(ResolvedFunction::Erlang)
                }
            }
        }
    }

    pub fn lookup_function_idx(
        &self,
        module: Atom,
        index: FunctionIndex,
    ) -> Option<&ErlangFunction> {
        let ret = match self.map.get(&module) {
            None => None,
            Some(ModuleType::Erlang(erl)) => Some(&erl.funs[&index]),
            Some(ModuleType::Overlayed(erl, _)) => Some(&erl.funs[&index]),
            Some(ModuleType::Native(_)) => unreachable!(),
        };

        if let Some(erl) = ret.as_ref() {
            trace!("LOOKUP IDX {}", erl.fun.ident());
        }

        ret
    }
}

#[derive(Copy, Clone)]
pub enum NativeFunctionKind {
    Simple(fn(&Arc<Process>, &[Term]) -> std::result::Result<Term, Exception>),
    Yielding(fn(&Arc<Process>, &[Term]) -> Result),
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
        fun: fn(&Arc<Process>, &[Term]) -> std::result::Result<Term, Exception>,
    ) {
        self.functions
            .insert((name, arity), NativeFunctionKind::Simple(fun));
    }

    pub fn add_yielding(
        &mut self,
        name: Atom,
        arity: usize,
        fun: fn(&Arc<Process>, &[Term]) -> Result,
    ) {
        self.functions
            .insert((name, arity), NativeFunctionKind::Yielding(fun));
    }
}

pub struct ErlangFunction {
    pub fun: Function,
    pub index: FunctionIndex,
    pub live: LiveValues,
}

pub struct ErlangModule {
    pub name: Atom,
    pub funs: BTreeMap<FunctionIndex, ErlangFunction>,
    pub name_map: BTreeMap<(Atom, usize), FunctionIndex>,
}

impl ErlangModule {
    pub fn from_eir(module: Module) -> Self {
        let name_atom = Atom::try_from_str(module.name().as_str()).unwrap();

        let funs = module
            .function_iter()
            .map(|fun_def| {
                let fun = fun_def.function();
                let nfun = ErlangFunction {
                    live: fun.live_values(),
                    index: fun_def.index(),
                    fun: fun.clone(),
                };
                (fun_def.index(), nfun)
            })
            .collect();

        let name_map = module
            .function_iter()
            .map(|fun_def| {
                let fun = fun_def.function();
                let ident = fun.ident();
                let name = Atom::try_from_str(ident.name.as_str()).unwrap();
                ((name, ident.arity), fun_def.index())
            })
            .collect();

        ErlangModule {
            name: name_atom,
            funs,
            name_map,
        }
    }
}

pub enum ModuleType {
    Erlang(ErlangModule),
    Overlayed(ErlangModule, NativeModule),
    Native(NativeModule),
}
