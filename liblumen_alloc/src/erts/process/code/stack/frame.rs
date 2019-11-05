use core::fmt::{self, Debug};

use alloc::sync::Arc;

use crate::erts::process::code::Code;
use crate::erts::term::Atom;
use crate::erts::term::Definition;
use crate::erts::ModuleFunctionArity;

pub struct Frame {
    definition: Definition,
    module_function_arity: Arc<ModuleFunctionArity>,
    code: Code,
}

impl Frame {
    pub fn new(module_function_arity: Arc<ModuleFunctionArity>, code: Code) -> Frame {
        Frame {
            definition: Definition::Export {
                function: module_function_arity.function,
            },
            module_function_arity,
            code,
        }
    }

    pub fn from_definition(module: Atom, definition: Definition, arity: u8, code: Code) -> Frame {
        Frame {
            module_function_arity: Arc::new(ModuleFunctionArity {
                module,
                function: definition.function_name(),
                arity,
            }),
            definition,
            code,
        }
    }

    pub fn module_function_arity(&self) -> Arc<ModuleFunctionArity> {
        Arc::clone(&self.module_function_arity)
    }

    pub fn definition<'a>(&'a self) -> &'a Definition {
        &self.definition
    }

    pub fn code(&self) -> Code {
        self.code
    }
}

impl Debug for Frame {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Frame")
            .field("module_function_arity", &self.module_function_arity)
            .field("code", &(self.code as *const ()))
            .finish()
    }
}

pub enum Placement {
    Replace,
    Push,
}
