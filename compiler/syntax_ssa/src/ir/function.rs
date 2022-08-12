use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt;
use std::rc::Rc;

use cranelift_entity::{entity_impl, PrimaryMap};

use liblumen_diagnostics::{SourceSpan, Spanned};
use liblumen_syntax_base::{FunctionName, Signature};

use super::*;

/// Represents the structure of a function
#[derive(Clone, Spanned)]
pub struct Function {
    pub id: FuncRef,
    #[span]
    pub span: SourceSpan,
    pub signature: Signature,
    pub dfg: DataFlowGraph,
}
impl fmt::Debug for Function {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Function(id = {:?}, span = {:?}, sig = {:?})",
            self.id, self.span, &self.signature
        )
    }
}
impl Function {
    pub fn new(
        id: FuncRef,
        span: SourceSpan,
        signature: Signature,
        signatures: Rc<RefCell<PrimaryMap<FuncRef, Signature>>>,
        callees: Rc<RefCell<BTreeMap<FunctionName, FuncRef>>>,
        constants: Rc<RefCell<ConstantPool>>,
    ) -> Self {
        let dfg = DataFlowGraph::new(signatures, callees, constants);
        Self {
            id,
            span,
            signature,
            dfg,
        }
    }
}

/// A handle that refers to a function either imported/local, or external
#[derive(Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FuncRef(u32);
entity_impl!(FuncRef, "fn");
