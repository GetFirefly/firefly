use liblumen_diagnostics::Spanned;
use liblumen_syntax_base::*;

use super::Fun;

#[derive(Debug, Clone, Spanned)]
pub struct Function {
    pub var_counter: usize,
    #[span]
    pub fun: Fun,
}
impl Eq for Function {}
impl PartialEq for Function {
    fn eq(&self, other: &Self) -> bool {
        self.fun == other.fun
    }
}
impl Annotated for Function {
    #[inline]
    fn annotations(&self) -> &Annotations {
        self.fun.annotations()
    }

    #[inline]
    fn annotations_mut(&mut self) -> &mut Annotations {
        self.fun.annotations_mut()
    }
}
