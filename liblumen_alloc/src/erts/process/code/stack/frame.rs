use alloc::sync::Arc;

#[cfg(debug_assertions)]
use core::fmt::{self, Debug};

use crate::erts::process::code::Code;
use crate::erts::ModuleFunctionArity;

pub struct Frame {
    module_function_arity: Arc<ModuleFunctionArity>,
    code: Code,
}

impl Frame {
    pub fn new(module_function_arity: Arc<ModuleFunctionArity>, code: Code) -> Frame {
        Frame {
            module_function_arity,
            code,
        }
    }

    pub fn module_function_arity(&self) -> Arc<ModuleFunctionArity> {
        Arc::clone(&self.module_function_arity)
    }

    pub fn code(&self) -> Code {
        self.code
    }
}

#[cfg(debug_assertions)]
impl Debug for Frame {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{{")?;
        writeln!(
            f,
            "  let frame = Frame::new({:?}, ...);",
            self.module_function_arity
        )?;

        writeln!(f, "  frame")?;
        write!(f, "}}")
    }
}
