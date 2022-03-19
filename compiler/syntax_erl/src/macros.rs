/// Construct a keyword list AST from a list of expressions, e.g.:
///
///     kwlist!(tuple!(atom!(key), var!(Value)))
///
/// It is required that elements are two-tuples, but no checking is done
/// to make sure that the first element in each tuple is an atom
#[allow(unused_macros)]
macro_rules! kwlist {
    ($($element:expr),*) => {
        let mut elements = vec![$($element),*];
        elements.reverse();
        elements.fold(nil!(), |acc, (key, val)| {
            cons!(tuple!(key, val), acc)
        })
    }
}

/// Like `kwlist!`, but produces a list of elements produced by any arbitrary expression
///
///     list!(atom!(foo), tuple!(atom!(bar), atom!(baz)))
///
/// Like `kwlist!`, this produces a proper list
#[allow(unused_macros)]
macro_rules! list {
    ($($element:expr),*) => {
        {
            let elements = [$($element),*];
            elements.iter().rev().fold(nil!(), |acc, el| {
                cons!(*el, acc)
            })
        }
    }
}

/// A lower-level primitive for constructing lists, via cons cells.
/// Given the following:
///
///     cons!(atom!(a), cons!(atom!(b), nil!()))
///
/// This is equivalent to `[a | [b | []]]`, which is in turn equivalent
/// to `[a, b]`. You are better off using `list!` unless you explicitly
/// need to construct an improper list
#[allow(unused_macros)]
macro_rules! cons {
    ($head:expr, $tail:expr) => {
        crate::ast::Expr::Cons(crate::ast::Cons {
            span: liblumen_diagnostics::SourceSpan::default(),
            head: Box::new($head),
            tail: Box::new($tail),
        })
    };
}

macro_rules! nil {
    () => {
        crate::ast::Expr::Nil(crate::ast::Nil(liblumen_diagnostics::SourceSpan::default()))
    };
}

/// Produces a tuple expression with the given elements
macro_rules! tuple {
    ($($element:expr),*) => {
        crate::ast::Expr::Tuple(crate::ast::Tuple{
            span: liblumen_diagnostics::SourceSpan::default(),
            elements: vec![$($element),*],
        })
    }
}

/// Produces an integer literal expression
macro_rules! int {
    ($i:expr) => {
        crate::ast::Expr::Literal(crate::ast::Literal::Integer(
            liblumen_diagnostics::SourceSpan::default(),
            $i,
        ))
    };
}

/// Produces a literal expression which evaluates to an atom
macro_rules! atom {
    ($sym:ident) => {
        crate::ast::Expr::Literal(crate::ast::Literal::Atom(
            liblumen_intern::Ident::with_empty_span(liblumen_intern::Symbol::intern(stringify!(
                $sym
            ))),
        ))
    };
    ($sym:expr) => {
        crate::ast::Expr::Literal(crate::ast::Literal::Atom(
            liblumen_intern::Ident::with_empty_span(liblumen_intern::Symbol::intern($sym)),
        ))
    };
}

macro_rules! atom_from_ident {
    ($id:expr) => {
        crate::ast::Expr::Literal(crate::ast::Literal::Atom($id))
    };
}

macro_rules! atom_from_sym {
    ($sym:expr) => {
        crate::ast::Expr::Literal(crate::ast::Literal::Atom(
            liblumen_intern::Ident::with_empty_span($sym),
        ))
    };
}

/// Produces an Ident from an expression, meant to be used to simplify generating
/// identifiers in the AST from strings or symbols
macro_rules! ident {
    ($sym:ident) => {
        liblumen_intern::Ident::with_empty_span(liblumen_intern::Symbol::intern(stringify!($sym)))
    };
    ($sym:expr) => {
        liblumen_intern::Ident::with_empty_span(liblumen_intern::Symbol::intern($sym))
    };
    (_) => {
        liblumen_intern::Ident::with_empty_span(liblumen_intern::Symbol::intern("_"))
    };
}

/// Produces an Option<Ident> from an expression, meant to be used to simplify generating
/// identifiers in the AST from strings or symbols
#[allow(unused_macros)]
macro_rules! ident_opt {
    ($sym:ident) => {
        Some(liblumen_intern::Ident::with_empty_span(
            liblumen_intern::Symbol::intern(stringify!($sym)),
        ))
    };
    ($sym:expr) => {
        Some(liblumen_intern::Ident::with_empty_span(
            liblumen_intern::Symbol::intern($sym),
        ))
    };
    (_) => {
        Some(liblumen_intern::Ident::with_empty_span(
            liblumen_intern::Symbol::intern("_"),
        ))
    };
}

/// Produces a variable expression
macro_rules! var {
    ($name:ident) => {
        crate::ast::Expr::Var(crate::ast::Var(ident!(stringify!($name))))
    };
    (_) => {
        crate::ast::Expr::Var(crate::ast::Var(ident!(_)))
    };
}

/// Produces a remote expression, e.g. `erlang:get_module_info`
///
/// Expects the module/function to be identifier symbols
macro_rules! remote {
    ($module:ident, $function:ident) => {
        crate::ast::Expr::Remote(crate::ast::Remote {
            span: liblumen_diagnostics::SourceSpan::default(),
            module: Box::new(atom!($module)),
            function: Box::new(atom!($function)),
        })
    };
    ($module:expr, $function:expr) => {
        crate::ast::Expr::Remote(crate::ast::Remote {
            span: liblumen_diagnostics::SourceSpan::default(),
            module: Box::new($module),
            function: Box::new($function),
        })
    };
}

/// Produces a function application expression
macro_rules! apply {
    ($callee:expr, $($args:expr),*) => {
        crate::ast::Expr::Apply(crate::ast::Apply {
            span: liblumen_diagnostics::SourceSpan::default(),
            callee: Box::new($callee),
            args: vec![$($args),*]
        })
    }
}

/// Produces a function definition
macro_rules! fun {
    ($name:ident ($($params:ident),*) -> $body:expr) => {
        {
            let params = vec![$(var!($params)),*];
            let arity = params.len().try_into().unwrap();
            crate::ast::Function {
                span: liblumen_diagnostics::SourceSpan::default(),
                name: ident!($name),
                arity,
                clauses: vec![
                    crate::ast::FunctionClause{
                        span: liblumen_diagnostics::SourceSpan::default(),
                        name: Some(crate::ast::Name::Atom(ident!($name))),
                        params,
                        guard: None,
                        body: vec![$body],
                    }
                ],
                spec: None,
            }
        }
    };
    ($name:ident $(($($params:expr),*) -> $body:expr);*) => {
        {
            let mut clauses = Vec::new();
            $(
                clauses.push(crate::ast::FunctionClause {
                    span: liblumen_diagnostics::SourceSpan::default(),
                    name: Some(crate::ast::Name::Atom(ident!($name))),
                    params: vec![$($params),*],
                    guard: None,
                    body: vec![$body],
                });
            )*
            let arity = clauses.first().as_ref().unwrap().params.len().try_into().unwrap();
            crate::ast::Function {
                span: liblumen_diagnostics::SourceSpan::default(),
                name: ident!($name),
                arity,
                clauses,
                spec: None,
            }
        }
    }
}
