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
    };
}

#[allow(unused_macros)]
macro_rules! kwlist_with_span {
    ($span:expr, $($element:expr),*) => {
        let mut elements = vec![$($element),*];
        elements.reverse();
        elements.fold(nil!(), |acc, (key, val)| {
            cons!($span, tuple_with_span!($span, key, val), acc)
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
    };
}

#[allow(unused_macros)]
macro_rules! list_with_span {
    ($span:expr, $($element:expr),*) => {
        {
            let elements = [$($element),*];
            elements.iter().rev().fold(nil!($span), |acc, el| {
                cons!($span, *el, acc)
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

    ($span:expr, $head:expr, $tail:expr) => {
        crate::ast::Expr::Cons(crate::ast::Cons {
            span: $span,
            head: Box::new($head),
            tail: Box::new($tail),
        })
    };
}

#[allow(unused_macros)]
macro_rules! nil {
    () => {
        crate::ast::Expr::Literal(crate::ast::Literal::Nil(
            liblumen_diagnostics::SourceSpan::default(),
        ))
    };

    ($span:expr) => {
        crate::ast::Expr::Literal(crate::ast::Literal::Nil($span))
    };
}

/// Produces a tuple expression with the given elements
#[allow(unused_macros)]
macro_rules! tuple {
    ($($element:expr),*) => {
        crate::ast::Expr::Tuple(crate::ast::Tuple{
            span: liblumen_diagnostics::SourceSpan::default(),
            elements: vec![$($element),*],
        })
    };
}

#[allow(unused_macros)]
macro_rules! tuple_with_span {
    ($span:expr, $($element:expr),*) => {
        crate::ast::Expr::Tuple(crate::ast::Tuple{
            span: $span,
            elements: vec![$($element),*],
        })
    };
}

/// Produces an integer literal expression
#[allow(unused_macros)]
macro_rules! int {
    ($i:expr) => {
        crate::ast::Expr::Literal(crate::ast::Literal::Integer(
            liblumen_diagnostics::SourceSpan::default(),
            $i,
        ))
    };

    ($span:expr, $i:expr) => {
        crate::ast::Expr::Literal(crate::ast::Literal::Integer($span, $i))
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
            liblumen_intern::Ident::with_empty_span($sym),
        ))
    };

    ($span:expr, $sym:ident) => {
        crate::ast::Expr::Literal(crate::ast::Literal::Atom(liblumen_intern::Ident::new(
            liblumen_intern::Symbol::intern(stringify!($sym)),
            $span,
        )))
    };

    ($span:expr, $sym:expr) => {
        crate::ast::Expr::Literal(crate::ast::Literal::Atom(liblumen_intern::Ident::new(
            $sym, $span,
        )))
    };
}

#[allow(unused_macros)]
macro_rules! atom_from_ident {
    ($id:expr) => {
        crate::ast::Expr::Literal(crate::ast::Literal::Atom($id))
    };
}

/// Produces an Ident from an expression, meant to be used to simplify generating
/// identifiers in the AST from strings or symbols
#[allow(unused_macros)]
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
#[allow(unused_macros)]
macro_rules! var {
    ($name:ident) => {
        crate::ast::Expr::Var(crate::ast::Var(ident!(stringify!($name))))
    };
    (_) => {
        crate::ast::Expr::Var(crate::ast::Var(ident!(_)))
    };
}

/// Produces a Symbol with the name of the given identifier
#[allow(unused_macros)]
macro_rules! symbol {
    ($sym:ident) => {
        liblumen_intern::Symbol::intern(stringify!($sym))
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
    ($name:ident ($($params:ident),*) -> $body:expr) => {
        {
            let patterns = vec![$(var!($params)),*];
            let arity = patterns.len().try_into().unwrap();
            crate::ast::Function {
                span: liblumen_diagnostics::SourceSpan::default(),
                name: ident!($name),
                arity,
                clauses: vec![
                    (
                        Some(crate::ast::Name::Atom(ident!($name))),
                        crate::ast::Clause{
                            span: liblumen_diagnostics::SourceSpan::default(),
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
    ($name:ident $(($($params:expr),*) -> $body:expr);*) => {
        {
            let mut clauses = Vec::new();
            $(
                clauses.push((
                    Some(crate::ast::Name::Atom(ident!($name))),
                    crate::ast::Clause {
                    span: liblumen_diagnostics::SourceSpan::default(),
                    patterns: vec![$($params),*],
                    guards: vec![],
                    body: vec![$body],
                    compiler_generated: false,
                }));
            )*
            let arity = clauses.first().as_ref().map(|(_, c)| c.patterns.len()).unwrap().try_into().unwrap();
            crate::ast::Function {
                span: liblumen_diagnostics::SourceSpan::default(),
                name: ident!($name),
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
