mod translate;

pub use self::translate::{CoreToKernel, KernelToSsa};

use rpds::{rbt_set, RedBlackTreeMap, RedBlackTreeSet};

use firefly_diagnostics::SourceSpan;
use firefly_intern::{Ident, Symbol};
use firefly_syntax_base::{Annotations, FunctionName, Var};

use crate::{Expr, Function, Name};

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionContext {
    span: SourceSpan,
    var_counter: usize,
    fun_counter: usize,
    args: Vec<Var>,
    name: FunctionName,
    funs: Vec<Function>,
    defined: RedBlackTreeSet<Ident>,
    free: RedBlackTreeMap<Name, Vec<Expr>>,
    labels: RedBlackTreeSet<Symbol>,
    ignore_funs: bool,
}
impl FunctionContext {
    pub fn new(span: SourceSpan, name: FunctionName, var_counter: usize) -> Self {
        Self {
            span,
            var_counter,
            fun_counter: 0,
            args: vec![],
            name,
            funs: vec![],
            defined: rbt_set![],
            free: RedBlackTreeMap::new(),
            labels: rbt_set![],
            ignore_funs: false,
        }
    }

    pub fn new_fun_name(&mut self, ty: Option<&str>) -> Symbol {
        let ty = ty.unwrap_or("anonymous");
        let name = format!(
            "-{}/{}-{}-{}-",
            self.name.function, self.name.arity, ty, self.fun_counter
        );
        self.fun_counter += 1;
        Symbol::intern(&name)
    }

    pub fn next_var_name(&mut self, span: Option<SourceSpan>) -> Ident {
        let id = self.var_counter;
        self.var_counter += 1;
        let var = format!("${}", id);
        let mut ident = Ident::from_str(&var);
        if let Some(span) = span {
            ident.span = span;
        }
        ident
    }

    pub fn next_var(&mut self, span: Option<SourceSpan>) -> Var {
        let name = self.next_var_name(span);
        Var {
            annotations: Annotations::default_compiler_generated(),
            name,
            arity: None,
        }
    }

    pub fn new_vars<const N: usize>(&mut self, span: Option<SourceSpan>) -> [Var; N] {
        let vars = [0; N];
        vars.map(|_| self.next_var(span))
    }

    pub fn n_vars(&mut self, n: usize, span: Option<SourceSpan>) -> Vec<Var> {
        (0..n).map(|_| self.next_var(span)).collect()
    }
}
