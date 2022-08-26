#[macro_export]
macro_rules! iatom {
    ($span:expr, $sym:expr) => {
        firefly_syntax_core::IExpr::Literal(lit_atom!($span, $sym))
    };
}

#[macro_export]
macro_rules! iint {
    ($span:expr, $i:expr) => {
        firefly_syntax_core::IExpr::Literal(lit_int!($span, $i))
    };
}

#[macro_export]
macro_rules! ituple {
    ($span:expr, $($element:expr),*) => {
        firefly_syntax_core::IExpr::Tuple(firefly_syntax_core::ITuple::new($span, vec![$($element),*]))
    };
}

#[macro_export]
macro_rules! icons {
    ($span:expr, $head:expr, $tail:expr) => {
        firefly_syntax_core::IExpr::Cons(firefly_syntax_core::ICons::new($span, $head, $tail))
    };
}

#[macro_export]
macro_rules! inil {
    ($span:expr) => {
        firefly_syntax_core::IExpr::Literal(lit_nil!($span))
    };
}

#[macro_export]
macro_rules! icall_eq_true {
    ($span:expr, $v:expr) => {{
        let span = $span;
        firefly_syntax_core::IExpr::Call(firefly_syntax_core::ICall {
            span,
            annotations: firefly_syntax_base::Annotations::default_compiler_generated(),
            module: Box::new(iatom!(span, firefly_intern::symbols::Erlang)),
            function: Box::new(iatom!(span, firefly_intern::symbols::EqualStrict)),
            args: vec![$v, iatom!(span, firefly_intern::symbols::True)],
        })
    }};
}
