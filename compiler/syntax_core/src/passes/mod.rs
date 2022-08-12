use liblumen_diagnostics::SourceSpan;
use liblumen_intern::{Ident, Symbol};
use liblumen_syntax_base::*;

mod annotate;
mod known;
mod rewrites;

pub use self::annotate::AnnotateVariableUsage;
pub(self) use self::known::Known;
pub use self::rewrites::*;

#[derive(Debug, PartialEq)]
pub struct FunctionContext {
    pub span: SourceSpan,
    pub var_counter: usize,
    pub fun_counter: usize,
    pub goto_counter: usize,
    pub name: FunctionName,
    pub wanted: bool,
    pub in_guard: bool,
    pub is_nif: bool,
}

impl FunctionContext {
    pub fn new(
        span: SourceSpan,
        name: FunctionName,
        var_counter: usize,
        fun_counter: usize,
    ) -> Self {
        Self {
            span,
            var_counter,
            fun_counter,
            goto_counter: 0,
            name,
            wanted: true,
            in_guard: false,
            is_nif: false,
        }
    }

    #[inline]
    pub fn set_wanted(&mut self, wanted: bool) -> bool {
        let prev = self.wanted;
        self.wanted = wanted;
        prev
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

    pub fn next_n_vars(&mut self, n: usize, span: Option<SourceSpan>) -> Vec<Var> {
        (0..n).map(|_| self.next_var(span)).collect()
    }

    pub fn new_fun_name(&mut self, ty: Option<&str>) -> Symbol {
        let name = if let Some(ty) = ty {
            format!("{}$^{}", ty, self.fun_counter)
        } else {
            format!(
                "-{}/{}-fun-{}-",
                self.name.function, self.name.arity, self.fun_counter
            )
        };
        self.fun_counter += 1;
        Symbol::intern(&name)
    }

    pub fn goto_func(&self) -> Var {
        let sym = Symbol::intern(&format!("label^{}", self.goto_counter));
        Var::new_with_arity(Ident::with_empty_span(sym), 0)
    }

    pub fn inc_goto_func(&mut self) {
        self.goto_counter += 1;
    }
}
