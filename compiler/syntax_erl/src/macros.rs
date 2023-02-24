/// Construct a keyword list AST from a list of expressions, e.g.:
///
/// It is required that elements are two-tuples, but no checking is done
/// to make sure that the first element in each tuple is an atom
#[allow(unused_macros)]
macro_rules! kwlist {
    ($span:expr, $($element:expr),*) => {
        let mut elements = vec![$($element),*];
        elements.reverse();
        elements.fold(nil!($span), |acc, (key, val)| {
            cons!($span, tuple!($span, key, val), acc)
        })
    };
}

/// Like `kwlist!`, but produces a list of elements produced by any arbitrary expression
#[allow(unused_macros)]
macro_rules! list {
    ($span:expr, $($element:expr),*) => {
        {
            let elements = [$($element),*];
            elements.iter().rev().fold(nil!($span), |acc, el| {
                cons!($span, *el, acc)
            })
        }
    };
}

macro_rules! ast_lit_list {
    ($span:expr, $($element:expr),*) => {
        {
            use smallvec::smallvec_inline;
            let mut elements = smallvec_inline![$($element),*];
            elements.drain(..).rev().fold(ast_lit_nil!($span), |tail, head| {
                ast_lit_cons!($span, head, tail)
            })
        }
    };
}

/// A lower-level primitive for constructing lists, via cons cells.
#[allow(unused_macros)]
macro_rules! cons {
    ($span:expr, $head:expr, $tail:expr) => {
        crate::ast::Expr::Cons(crate::ast::Cons {
            span: $span,
            head: Box::new($head),
            tail: Box::new($tail),
        })
    };
}

macro_rules! ast_lit_cons {
    ($span:expr, $head:expr, $tail:expr) => {
        crate::ast::Literal::Cons($span, Box::new($head), Box::new($tail))
    };
}

#[allow(unused_macros)]
macro_rules! nil {
    ($span:expr) => {
        crate::ast::Expr::Literal(crate::ast::Literal::Nil($span))
    };
}

#[allow(unused_macros)]
macro_rules! ast_lit_nil {
    ($span:expr) => {
        crate::ast::Literal::Nil($span)
    };
}

/// Produces a tuple expression with the given elements
#[allow(unused_macros)]
macro_rules! tuple {
    ($span:expr, $($element:expr),*) => {
        crate::ast::Expr::Tuple(crate::ast::Tuple{
            span: $span,
            elements: vec![$($element),*],
        })
    };
}

#[allow(unused_macros)]
macro_rules! ast_lit_tuple {
    ($span:expr, $($element:expr),*) => {
        crate::ast::Literal::Tuple($span, vec![$($element),*])
    };
}

/// Produces an integer literal expression
#[allow(unused_macros)]
macro_rules! int {
    ($span:expr, $i:expr) => {
        crate::ast::Expr::Literal(crate::ast::Literal::Integer($span, $i))
    };
}

macro_rules! ast_lit_int {
    ($span:expr, $i:expr) => {
        crate::ast::Literal::Integer($span, $i)
    };
}

macro_rules! ast_lit_atom {
    ($span:expr, $sym:expr) => {
        crate::ast::Literal::Atom(firefly_intern::Ident::new($sym, $span))
    };
}

/// Produces a literal expression which evaluates to an atom
macro_rules! atom {
    ($span:expr, $sym:ident) => {
        crate::ast::Expr::Literal(crate::ast::Literal::Atom(firefly_intern::Ident::new(
            firefly_intern::Symbol::intern(stringify!($sym)),
            $span,
        )))
    };

    ($span:expr, $sym:expr) => {
        crate::ast::Expr::Literal(crate::ast::Literal::Atom(firefly_intern::Ident::new(
            $sym, $span,
        )))
    };
}

macro_rules! atom_from_ident {
    ($id:expr) => {
        crate::ast::Expr::Literal(crate::ast::Literal::Atom($id))
    };
}

/// Produces an Ident from an expression, meant to be used to simplify generating
/// identifiers in the AST from strings or symbols
macro_rules! ident {
    ($span:expr, $sym:ident) => {
        firefly_intern::Ident::new(firefly_intern::Symbol::intern(stringify!($sym)), $span)
    };
    ($span:expr, $sym:expr) => {
        firefly_intern::Ident::new(firefly_intern::Symbol::intern($sym), $span)
    };
    ($span:expr, _) => {
        firefly_intern::Ident::new(firefly_intern::Symbol::intern("_"), $span)
    };
}

/// Produces an Option<Ident> from an expression, meant to be used to simplify generating
/// identifiers in the AST from strings or symbols
#[allow(unused_macros)]
macro_rules! ident_opt {
    ($span:expr, $sym:ident) => {
        Some(ident!($span, $sym))
    };
    ($span:expr, $sym:expr) => {
        Some(ident!($span, $sym))
    };
    ($span:expr, _) => {
        Some(ident!($span, _))
    };
}

/// Produces a variable expression
#[allow(unused_macros)]
macro_rules! var {
    ($span:expr, $name:ident) => {
        crate::ast::Expr::Var(crate::ast::Var(ident!($span, stringify!($name))))
    };
    ($span:expr, _) => {
        crate::ast::Expr::Var(crate::ast::Var(ident!($span, _)))
    };
}

/// Produces a Symbol with the name of the given identifier
#[allow(unused_macros)]
macro_rules! symbol {
    ($sym:ident) => {
        firefly_intern::Symbol::intern(stringify!($sym))
    };
}

/// Produces a function application expression
#[allow(unused_macros)]
macro_rules! apply {
    ($span:expr, $module:ident, $function:ident, ($($args:expr),*)) => {
        {
            let span = $span;
            let args = vec![$($args),*];
            crate::ast::Expr::Apply(crate::ast::Apply::remote(span, symbol!($module), symbol!($function), args))
        }
    };

    ($span:expr, $module:ident, $function:expr, ($($args:expr),*)) => {
        {
            let span = $span;
            let args = vec![$($args),*];
            crate::ast::Expr::Apply(crate::ast::Apply::remote(span, symbol!($module), $function, args))
        }
    };

    ($span:expr, $module:expr, $function:expr, ($($args:expr),*)) => {
        {
            let span = $span;
            let args = vec![$($args),*];
            crate::ast::Expr::Apply(crate::ast::Apply::remote(span, $module, $function, args))
        }
    };

    ($span:expr, $callee:expr, ($($args:expr),*)) => {
        {
            let span = $span;
            let args = vec![$($args),*];
            crate::ast::Expr::Apply(crate::ast::Apply::new(span, $callee, args))
        }
    };
}

/// Produces a function definition
#[allow(unused_macros)]
macro_rules! fun {
    ($span:expr, $name:ident ($($params:ident),*) -> $body:expr) => {
        {
            let patterns = vec![$(var!($span, $params)),*];
            let arity = patterns.len().try_into().unwrap();
            crate::ast::Function {
                span: $span,
                name: ident!($span, $name),
                arity,
                clauses: vec![
                    (
                        Some(crate::ast::Name::Atom(ident!($span, $name))),
                        crate::ast::Clause{
                            span: $span,
                            patterns,
                            guards: vec![],
                            body: vec![$body],
                            compiler_generated: false,
                        }
                    )
                ],
                spec: None,
                is_nif: false,
                var_counter: 0,
                fun_counter: 0,
            }
        }
    };
    ($span:expr, $name:ident $(($($params:expr),*) -> $body:expr);*) => {
        {
            let mut clauses = Vec::new();
            $(
                clauses.push((
                    Some(crate::ast::Name::Atom(ident!($span, $name))),
                    crate::ast::Clause {
                    span: $span,
                    patterns: vec![$($params),*],
                    guards: vec![],
                    body: vec![$body],
                    compiler_generated: false,
                }));
            )*
            let arity = clauses.first().as_ref().map(|(_, c)| c.patterns.len()).unwrap().try_into().unwrap();
            crate::ast::Function {
                span: $span,
                name: ident!($span, $name),
                arity,
                clauses,
                spec: None,
                is_nif: false,
                var_counter: 0,
                fun_counter: 0,
            }
        }
    }
}
