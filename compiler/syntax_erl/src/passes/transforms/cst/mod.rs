///! Purpose : Transform normal Erlang to Core Erlang
///!
///! At this stage all preprocessing has been done. All that is left are
///! "pure" Erlang functions.
///!
///! Core transformation is done in four stages:
///!
///! 1. Flatten expressions into an internal core form without doing
///!    matching.
///!
///! 2. Step "forwards" over the icore code annotating each "top-level"
///!    thing with variable usage.  Detect bound variables in matching
///!    and replace with explicit guard test.  Annotate "internal-core"
///!    expressions with variables they use and create.  Convert matches
///!    to cases when not pure assignments.
///!
///! 3. Step "backwards" over icore code using variable usage
///!    annotations to change implicit exported variables to explicit
///!    returns.
///!
///! 4. Lower receives to more primitive operations.  Split binary
///!    patterns where a value is matched out and then used used as
///!    a size in the same pattern.  That simplifies the subsequent
///!    passes as all variables are within a single pattern are either
///!    new or used, but never both at the same time.
///!
///! To ensure the evaluation order we ensure that all arguments are
///! safe.  A "safe" is basically a core_lib simple with VERY restricted
///! binaries.
///!
///! We have to be very careful with matches as these create variables.
///! While we try not to flatten things more than necessary we must make
///! sure that all matches are at the top level.  For this we use the
///! type "novars" which are non-match expressions.  Cases and receives
///! can also create problems due to exports variables so they are not
///! "novars" either.  I.e. a novars will not export variables.
///!
///! Annotations in the #iset, #iletrec, and all other internal records
///! is kept in a record, #a, not in a list as in proper core.  This is
///! easier and faster and creates no problems as we have complete control
///! over all annotations.
///!
///! On output, the annotation for most Core Erlang terms will contain
///! the source line number. A few terms will be marked with the atom
///! atom 'compiler_generated', to indicate that the compiler has generated
///! them and that no warning should be generated if they are optimized
///! away.
///!
///!
///! In this translation:
///!
///! call ops are safes
///! call arguments are safes
///! match arguments are novars
///! case arguments are novars
///! receive timeouts are novars
///! binaries and maps are novars
///! let/set arguments are expressions
///! fun is not a safe
use std::cell::UnsafeCell;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::rc::Rc;

use liblumen_binary::{BinaryEntrySpecifier, BitVec};
use liblumen_diagnostics::*;
use liblumen_intern::{symbols, Ident, Symbol};
use liblumen_number::Integer;
use liblumen_pass::Pass;

use crate::ast;
use crate::ast::{BinaryOp, UnaryOp};
use crate::cst::{self, *};
use crate::evaluator;
use crate::{Arity, Name};

macro_rules! lit_atom {
    ($span:expr, $sym:expr) => {
        Literal::atom($span, $sym)
    };
}

macro_rules! lit_int {
    ($span:expr, $i:expr) => {
        Literal::integer($span, $i)
    };
}

macro_rules! lit_tuple {
    ($span:expr, $($element:expr),*) => {
        Literal::tuple($span, vec![$($element),*])
    };
}

macro_rules! lit_nil {
    ($span:expr) => {
        Literal::nil($span)
    };
}

macro_rules! catom {
    ($span:expr, $sym:expr) => {
        Expr::Literal(lit_atom!($span, $sym))
    };
}

macro_rules! cint {
    ($span:expr, $i:expr) => {
        Expr::Literal(lit_int!($span, $i))
    };
}

macro_rules! ctuple {
    ($span:expr, $($element:expr),*) => {
        Expr::Tuple(Tuple::new($span, vec![$($element),*]))
    };
}

macro_rules! ccons {
    ($span:expr, $head:expr, $tail:expr) => {
        Expr::Cons(Cons::new($span, $head, $tail))
    };
}

macro_rules! cnil {
    ($span:expr) => {
        Expr::Literal(lit_nil!($span))
    };
}

macro_rules! icall_eq_true {
    ($span:expr, $v:expr) => {{
        let span = $span;
        Expr::Call(Call {
            span,
            annotations: Annotations::default_compiler_generated(),
            module: Box::new(catom!(span, symbols::Erlang)),
            function: Box::new(catom!(span, symbols::EqualStrict)),
            args: vec![$v, catom!(span, symbols::True)],
        })
    }};
}

#[derive(Debug, PartialEq)]
struct FunctionContext {
    span: SourceSpan,
    var_counter: usize,
    fun_counter: usize,
    name: Ident,
    arity: u8,
    wanted: bool,
    in_guard: bool,
}

mod annotate;
mod lower;
mod rewrites;
mod simplify;

use self::annotate::AnnotateVarUsage;
use self::lower::LowerAst;
use self::rewrites::RewriteExports;
use self::simplify::SimplifyCst;

/// This pass transforms an AST function into its CST form for further analysis and eventual lowering to Core IR
///
/// This pass performs numerous small transformations to normalize the structure of the AST
pub struct AstToCst {
    reporter: Reporter,
}
impl AstToCst {
    pub fn new(reporter: Reporter) -> Self {
        Self { reporter }
    }
}
impl Pass for AstToCst {
    type Input<'a> = ast::Module;
    type Output<'a> = cst::Module;

    fn run<'a>(&mut self, mut ast: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        let mut module = Module {
            span: ast.span,
            annotations: Annotations::default(),
            name: ast.name,
            vsn: None,    // TODO
            author: None, // TODO
            compile: ast.compile,
            on_load: ast.on_load,
            nifs: ast.nifs,
            imports: ast.imports,
            exports: ast.exports,
            behaviours: ast.behaviours,
            attributes: ast
                .attributes
                .drain()
                .filter_map(|(k, v)| translate_attr(k, v.value))
                .collect(),
            functions: BTreeMap::new(),
        };

        while let Some((name, function)) = ast.functions.pop_first() {
            let context = Rc::new(UnsafeCell::new(FunctionContext::new(&function)));
            let mut pipeline = LowerAst::new(self.reporter.clone(), Rc::clone(&context))
                .chain(AnnotateVarUsage::new(Rc::clone(&context)))
                .chain(RewriteExports::new(Rc::clone(&context)))
                .chain(SimplifyCst::new(Rc::clone(&context)));
            let function = pipeline.run(function)?;
            module.functions.insert(name, function);
        }

        Ok(module)
    }
}

fn translate_attr(_name: Ident, _value: ast::Expr) -> Option<(Ident, Expr)> {
    // TODO:
    None
}

impl FunctionContext {
    fn new(f: &ast::Function) -> Self {
        Self {
            span: f.span,
            var_counter: f.var_counter,
            fun_counter: f.fun_counter,
            name: f.name,
            arity: f.arity,
            wanted: true,
            in_guard: false,
        }
    }

    #[inline]
    fn set_wanted(&mut self, wanted: bool) -> bool {
        let prev = self.wanted;
        self.wanted = wanted;
        prev
    }

    fn next_var(&mut self, span: Option<SourceSpan>) -> Var {
        let id = self.var_counter;
        self.var_counter += 1;
        let var = format!("${}", id);
        let mut ident = Ident::from_str(&var);
        if let Some(span) = span {
            ident.span = span;
        }
        Var {
            span: ident.span,
            annotations: Annotations::default_compiler_generated(),
            name: ident,
            arity: None,
        }
    }

    fn new_fun_name(&mut self, ty: Option<&str>) -> Symbol {
        let name = if let Some(ty) = ty {
            format!("{}$^{}", ty, self.fun_counter)
        } else {
            format!(
                "-{}/{}-fun-{}-",
                self.name.name, self.arity, self.fun_counter
            )
        };
        self.fun_counter += 1;
        Symbol::intern(&name)
    }
}

/// Here follows an abstract data structure to help us handle Erlang's
/// implicit matching that occurs when a variable is bound more than
/// once:
///
///     X = Expr1(),
///     X = Expr2()
///
/// What is implicit in Erlang, must be explicit in Core Erlang; that
/// is, repeated variables must be eliminated and explicit matching
/// must be added. For simplicity, examples that follow will be given
/// in Erlang and not in Core Erlang. Here is how the example can be
/// rewritten in Erlang to eliminate the repeated variable:
///
///     X = Expr1(),
///     X1 = Expr2(),
///     if
///         X1 =:= X -> X;
///         true -> error({badmatch,X1})
///     end
///
/// To implement the renaming, keeping a set of the variables that
/// have been bound so far is **almost** sufficient. When a variable
/// in the set is bound a again, it will be renamed and a `case` with
/// guard test will be added.
///
/// Here is another example:
///
///     (X=A) + (X=B)
///
/// Note that the operands for a binary operands are allowed to be
/// evaluated in any order. Therefore, variables bound on the left
/// hand side must not referenced on the right hand side, and vice
/// versa. If a variable is bound on both sides, it must be bound
/// to the same value.
///
/// Using the simple scheme of keeping track of known variables,
/// the example can be rewritten like this:
///
///     X = A,
///     X1 = B,
///     if
///         X1 =:= X -> ok;
///         true -> error({badmatch,X1})
///     end,
///     X + X1
///
/// However, this simple scheme of keeping all previously bound variables in
/// a set breaks down for this example:
///
///     (X=A) + fun() -> X = B end()
///
/// The rewritten code would be:
///
///     X = A,
///     Tmp = fun() ->
///               X1 = B,
///               if
///                   X1 =:= X -> ok;
///                   true -> error({badmatch,X1})
///               end
///           end(),
///     X + Tmp
///
/// That is wrong, because the binding of `X` created on the left hand
/// side of `+` must not be seen inside the fun. The correct rewrite
/// would be like this:
///
///     X = A,
///     Tmp = fun() ->
///               X1 = B
///           end(),
///     X + Tmp
///
/// To correctly rewrite fun bodies, we will need to keep addtional
/// information in a record so that we can remove `X` from the known
/// variables when rewriting the body of the fun.
///
#[derive(Clone, Default)]
struct Known {
    base: Vec<BTreeSet<Ident>>,
    ks: BTreeSet<Ident>,
    prev_ks: Vec<BTreeSet<Ident>>,
}
impl Known {
    /// Get the currently known variables
    fn get(&self) -> &BTreeSet<Ident> {
        &self.ks
    }

    fn upat_is_new_var(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Var(v) => self.ks.contains(&v.name),
            _ => false,
        }
    }

    fn start_group(&mut self) {
        self.prev_ks.push(BTreeSet::new());
        self.base.push(self.ks.clone());
    }

    fn end_body(&mut self) {
        self.prev_ks.pop();
        self.prev_ks.push(self.ks.clone());
    }

    /// known_end_group(#known{}) -> #known{}.
    ///  Consolidate the known variables after having processed the
    ///  last body in a group of bodies that see the same bindings.
    fn end_group(&mut self) {
        self.base.pop();
        self.prev_ks.pop();
    }

    /// known_union(#known{}, KnownVarsSet) -> #known{}.
    ///  Update the known variables to be the union of the previous
    ///  known variables and the set KnownVarsSet.
    fn union(&mut self, vars: &BTreeSet<Ident>) {
        let ks = self.ks.union(vars).cloned().collect();
        self.ks = ks;
    }

    /// known_bind(#known{}, BoundVarsSet) -> #known{}.
    ///  Add variables that are known to be bound in the current
    ///  body.
    fn bind(&mut self, vars: &BTreeSet<Ident>) {
        if let Some(prev) = self.prev_ks.pop() {
            self.prev_ks.push(prev.difference(vars).cloned().collect());
        }
    }

    /// known_in_fun(#known{}) -> #known{}.
    ///  Update the known variables to only the set of variables that
    ///  should be known when entering the fun.
    fn known_in_fun(&mut self) {
        if self.base.is_empty() || self.prev_ks.is_empty() {
            return;
        }

        // Within a group of bodies that see the same bindings, calculate
        // the known variables for a fun. Example:
        //
        //     A = 1,
        //     {X = 2, fun() -> X = 99, A = 1 end()}.
        //
        // In this example:
        //
        //     BaseKs = ['A'], Ks0 = ['A','X'], PrevKs = ['A','X']
        //
        // Thus, only `A` is known when entering the fun.
        let base_ks = self.base.pop().unwrap();
        let prev_ks = self.prev_ks.pop().unwrap();
        let diff = self.ks.difference(&prev_ks).cloned().collect();
        let ks = &base_ks & &diff;
        self.base = vec![];
        self.prev_ks = vec![];
        self.ks = ks;
    }
}
