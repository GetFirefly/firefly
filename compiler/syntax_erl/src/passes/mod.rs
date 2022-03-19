mod desugar;
mod lower;
mod sema;

pub use self::desugar::*;
pub use self::lower::AstToCore;
pub use self::sema::*;

use liblumen_diagnostics::SourceSpan;
use liblumen_intern::Ident;

/// This struct is used to represent state about the current top-level function
/// that a pass is operating on.
///
/// Some transformations care about the current function name/arity.
///
/// Many transformations rely on introducing new variable bindings, and in order to
/// ensure that no conflicts with existing bindings occur, a unique id generator is
/// needed for each function.
///
/// Any other state which is mutable, and function-scoped, and is not pass-specific,
/// will need to be stored in this struct.
///
/// The struct itself is not used directly, but is instead wrapped in an `Rc<RefCell<T>>`
/// so that multiple passes can be instantiated at once with a reference to it, while
/// preserving the ability of those passes to mutate the underlying struct. Passes should
/// only borrow the struct at the specific times where data needs to be accessed, and avoid
/// at all costs borrowing across the execution of other passes which also hold a reference,
/// as a mutable borrow in such a situation will result in a panic. The pass infra is
/// designed such that this cannot happen unless you specifically borrow the struct while
/// simultaneously executing a child pass that also holds a reference to the context, so
/// it should be easily caught during code review.
#[derive(Debug)]
pub struct FunctionContext {
    pub name: Ident,
    pub arity: u8,
    pub var_counter: usize,
}
impl FunctionContext {
    #[inline]
    pub fn new(name: Ident, arity: u8) -> Self {
        Self {
            name,
            arity,
            var_counter: 0,
        }
    }

    pub fn next_var(&mut self, span: Option<SourceSpan>) -> Ident {
        let id = self.var_counter;
        self.var_counter += 1;
        let var = format!("${}", id);
        let mut ident = Ident::from_str(&var);
        match span {
            None => ident,
            Some(span) => {
                ident.span = span;
                ident
            }
        }
    }
}
