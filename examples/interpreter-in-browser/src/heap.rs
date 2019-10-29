use core::ptr::NonNull;

use wasm_bindgen::prelude::*;

use liblumen_alloc::atom;
use liblumen_alloc::erts::process::HeapAlloc;
use liblumen_alloc::erts::term;
use liblumen_alloc::erts::term::prelude::{Term, Atom, Encode};
use liblumen_alloc::erts::HeapFragment;
use liblumen_alloc::erts::ModuleFunctionArity;
use lumen_runtime::process::spawn::options::Options;
use lumen_runtime::registry::pid_to_process;
use lumen_runtime::scheduler::{Scheduler, Spawned};

#[wasm_bindgen]
pub struct Pid(term::prelude::Pid);

#[wasm_bindgen]
pub struct JsHeap {
    fragment: NonNull<HeapFragment>,
    terms: Vec<Term>,
}

#[wasm_bindgen]
impl JsHeap {
    #[wasm_bindgen(constructor)]
    pub fn new(size: usize) -> JsHeap {
        let fragment = unsafe { HeapFragment::new_from_word_size(size) }.unwrap();
        JsHeap {
            fragment,
            terms: Vec::new(),
        }
    }

    fn push(&mut self, term: Term) -> usize {
        let idx = self.terms.len();
        self.terms.push(term);
        idx
    }

    pub fn atom(&mut self, name: &str) -> usize {
        self.push(atom!(name))
    }

    pub fn integer(&mut self, number: i32) -> usize {
        let frag = unsafe { self.fragment.as_mut() };
        let term = frag.integer(number).unwrap();
        self.push(term)
    }

    pub fn tuple(&mut self, elems: &[usize]) -> usize {
        let frag = unsafe { self.fragment.as_mut() };
        let terms = &self.terms;
        let term = frag
            .tuple_from_iter(elems.iter().map(|n| terms[*n]), elems.len())
            .unwrap();
        self.push(term.encode().unwrap())
    }

    pub fn send(&self, pid: Pid, msg: usize) {
        match pid_to_process(&pid.0) {
            Some(process) => {
                let term = self.terms[msg];
                process.send_from_other(term).unwrap();
            }
            None => (),
        }
    }

    pub fn spawn(&self, m: &str, f: &str, a: &[usize], heap_size: usize) -> Pid {
        let module = Atom::try_from_str(m).unwrap();
        let function = Atom::try_from_str(f).unwrap();

        let proc = liblumen_eir_interpreter::VM.init.clone();

        let return_ok = {
            let mfa = ModuleFunctionArity {
                module: Atom::try_from_str("lumen_eir_interpreter_intrinsics").unwrap(),
                function: Atom::try_from_str("return_ok").unwrap(),
                arity: 1,
            };
            proc.closure_with_env_from_slice(
                mfa.into(),
                liblumen_eir_interpreter::code::return_ok,
                proc.pid_term(),
                &[],
            )
            .unwrap()
        };
        let return_throw = {
            let mfa = ModuleFunctionArity {
                module: Atom::try_from_str("lumen_eir_interpreter_intrinsics").unwrap(),
                function: Atom::try_from_str("return_throw").unwrap(),
                arity: 3,
            };
            proc.closure_with_env_from_slice(
                mfa.into(),
                liblumen_eir_interpreter::code::return_throw,
                proc.pid_term(),
                &[],
            )
            .unwrap()
        };

        let mut args_vec = vec![return_ok, return_throw];
        args_vec.extend(a.iter().map(|v| self.terms[*v]));

        let arguments = proc.list_from_slice(&args_vec).unwrap();

        let mut options: Options = Default::default();
        options.min_heap_size = Some(heap_size);

        let Spawned {
            arc_process: run_arc_process,
            ..
        } = Scheduler::spawn_apply_3(&proc, options, module, function, arguments).unwrap();

        Pid(run_arc_process.pid())
    }
}
