use super::*;

/// Phase 3: Rewrite clauses to make implicit exports explicit
///
/// Step "backwards" over icore code using variable usage
/// annotations to change implicit exported variables to explicit
/// returns.
pub struct RewriteExports {
    context: Rc<UnsafeCell<FunctionContext>>,
}
impl RewriteExports {
    pub(super) fn new(context: Rc<UnsafeCell<FunctionContext>>) -> Self {
        Self { context }
    }

    #[inline(always)]
    fn context(&self) -> &FunctionContext {
        unsafe { &*self.context.get() }
    }

    #[inline(always)]
    fn context_mut(&self) -> &mut FunctionContext {
        unsafe { &mut *self.context.get() }
    }
}
impl Pass for RewriteExports {
    type Input<'a> = cst::IFun;
    type Output<'a> = cst::Fun;

    fn run<'a>(&mut self, _ifun: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        todo!()
    }
}
