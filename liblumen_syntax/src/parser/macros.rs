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
            cons!(Expr::Tuple(Tuple { span: ByteSpan::default(), elements: vec![key, val] }), acc)
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
            let mut elements = vec![$($element),*];
            elements.reverse();
            elements.iter().fold(nil!(), |acc, el| {
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
        Expr::Cons(Cons {
            span: ByteSpan::default(),
            head: Box::new($head),
            tail: Box::new($tail),
        })
    };
}

macro_rules! nil {
    () => {
        Expr::Nil(Nil(ByteSpan::default()))
    };
}

/// Produces a tuple expression with the given elements
macro_rules! tuple {
    ($($element:expr),*) => {
        Expr::Tuple(Tuple{
            span: ByteSpan::default(),
            elements: vec![$($element),*],
        })
    }
}

/// Produces an integer literal expression
macro_rules! int {
    ($i:expr) => {
        Expr::Literal(Literal::Integer(ByteSpan::default(), $i))
    };
}

/// Produces a literal expression which evaluates to an atom
macro_rules! atom {
    ($sym:ident) => {
        Expr::Literal(Literal::Atom(Ident::with_empty_span(Symbol::intern(
            stringify!($sym),
        ))))
    };
    ($sym:expr) => {
        Expr::Literal(Literal::Atom(Ident::with_empty_span(Symbol::intern($sym))))
    };
}

macro_rules! atom_from_sym {
    ($sym:expr) => {
        Expr::Literal(Literal::Atom(Ident::with_empty_span($sym)))
    };
}

/// Produces an Ident from an expression, meant to be used to simplify generating
/// identifiers in the AST from strings or symbols
macro_rules! ident {
    ($sym:ident) => {
        Ident::with_empty_span(Symbol::intern(stringify!($sym)))
    };
    ($sym:expr) => {
        Ident::with_empty_span(Symbol::intern($sym))
    };
    (_) => {
        Ident::with_empty_span(Symbol::intern("_"))
    };
}

/// Produces an Option<Ident> from an expression, meant to be used to simplify generating
/// identifiers in the AST from strings or symbols
#[allow(unused_macros)]
macro_rules! ident_opt {
    ($sym:ident) => {
        Some(Ident::with_empty_span(Symbol::intern(stringify!($sym))))
    };
    ($sym:expr) => {
        Some(Ident::with_empty_span(Symbol::intern($sym)))
    };
    (_) => {
        Ident::with_empty_span(Symbol::intern("_"))
    };
}

/// Produces a variable expression
macro_rules! var {
    ($name:ident) => {
        Expr::Var(ident!(stringify!($name)))
    };
    (_) => {
        Expr::Var(ident!(_))
    };
}

/// Produces a remote expression, e.g. `erlang:get_module_info`
///
/// Expects the module/function to be identifier symbols
macro_rules! remote {
    ($module:ident, $function:ident) => {
        Expr::Remote(Remote {
            span: ByteSpan::default(),
            module: Box::new(atom!($module)),
            function: Box::new(atom!($function)),
        })
    };
    ($module:expr, $function:expr) => {
        Expr::Remote(Remote {
            span: ByteSpan::default(),
            module: Box::new($module),
            function: Box::new($function),
        })
    };
}

/// Produces a function application expression
macro_rules! apply {
    ($callee:expr, $($args:expr),*) => {
        Expr::Apply(Apply {
            span: ByteSpan::default(),
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
            let arity = params.len();
            NamedFunction {
                span: ByteSpan::default(),
                name: ident!($name),
                arity,
                clauses: vec![
                    FunctionClause{
                        span: ByteSpan::default(),
                        name: Some(ident!($name)),
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
                clauses.push(FunctionClause {
                    span: ByteSpan::default(),
                    name: Some(ident!($name)),
                    params: vec![$($params),*],
                    guard: None,
                    body: vec![$body],
                });
            )*
            let arity = clauses.first().as_ref().unwrap().params.len();
            NamedFunction {
                span: ByteSpan::default(),
                name: ident!($name),
                arity,
                clauses,
                spec: None,
            }
        }
    }
}
