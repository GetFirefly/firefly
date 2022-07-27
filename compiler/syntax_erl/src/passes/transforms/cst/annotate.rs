use super::*;

/// Phase 2: Annotate variable usage
///
/// Step "forwards" over the icore code annotating each "top-level"
/// thing with variable usage.  Detect bound variables in matching
/// and replace with explicit guard test.  Annotate "internal-core"
/// expressions with variables they use and create.  Convert matches
/// to cases when not pure assignments.
pub struct AnnotateVarUsage {
    context: Rc<UnsafeCell<FunctionContext>>,
}
impl AnnotateVarUsage {
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
impl Pass for AnnotateVarUsage {
    type Input<'a> = cst::IFun;
    type Output<'a> = cst::IFun;

    fn run<'a>(&mut self, mut ifun: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        match self.uexpr(IExpr::Fun(ifun), Known::default()) {
            Ok(IExpr::Fun(ifun)) => Ok(ifun),
            Ok(_) => panic!("unexpected iexpr result, expected ifun"),
            Err(reason) => Err(reason),
        }
    }
}

impl AnnotateVarUsage {
    fn ufun_clauses(&mut self, mut clauses: Vec<IClause>, known: Known) -> Vec<IClause> {
        clauses
            .drain(..)
            .map(|c| self.ufun_clause(c, known.clone()))
            .collect()
    }

    fn ufun_clause(&mut self, clause: IClause, known: Known) -> IClause {
        // Since variables in fun heads shadow previous variables
        // with the same name, we used to send an empty list as the
        // known variables when doing liveness analysis of the patterns
        // (in the upattern functions).
        //
        // With the introduction of expressions in size for binary
        // segments and in map keys, all known variables must be
        // available when analysing those expressions, or some variables
        // might not be seen as used if, for example, the expression includes
        // a case construct.
        //
        // Therefore, we will send in the complete list of known variables
        // when doing liveness analysis of patterns. This is
        // safe because any shadowing variables in a fun head has
        // been renamed.
        let (mut clause, pvs, used, _) = self.do_uclause(clause, known);
        let used: BTreeSet<Ident> = used.difference(&pvs).cloned().collect();
        clause.annotate(symbols::Used, used);
        clause
    }

    fn uclauses(&mut self, mut clauses: Vec<IClause>, known: Known) -> Vec<IClause> {
        clauses
            .drain(..)
            .map(|c| self.uclause(c, known.clone()))
            .collect()
    }

    fn uclause(&mut self, clause: IClause, known: Known) -> IClause {
        let (mut clause, _pv, used, new) = self.do_uclause(clause, known);
        clause.annotate(symbols::Used, used);
        clause.annotate(symbols::New, new);
        clause
    }

    fn do_uclause(
        &mut self,
        _clause: IClause,
        _known: Known,
    ) -> (IClause, BTreeSet<Ident>, BTreeSet<Ident>, BTreeSet<Ident>) {
        todo!()
    }

    fn uexpr(&mut self, expr: IExpr, _known: Known) -> anyhow::Result<IExpr> {
        dbg!(&expr);
        todo!()
    }
}
