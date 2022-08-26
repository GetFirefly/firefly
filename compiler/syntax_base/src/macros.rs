#[macro_export]
macro_rules! annotated {
    ($t:ident) => {
        impl Annotated for $t {
            fn annotations(&self) -> &Annotations {
                &self.annotations
            }

            fn annotations_mut(&mut self) -> &mut Annotations {
                &mut self.annotations
            }
        }
    };
}

#[macro_export]
macro_rules! lit_atom {
    ($span:expr, $sym:expr) => {
        firefly_syntax_base::Literal::atom($span, $sym)
    };
}

#[macro_export]
macro_rules! lit_int {
    ($span:expr, $i:expr) => {
        firefly_syntax_base::Literal::integer($span, $i)
    };
}

#[macro_export]
macro_rules! lit_tuple {
    ($span:expr, $($element:expr),*) => {
        firefly_syntax_base::Literal::tuple($span, vec![$($element),*])
    };
}

#[macro_export]
macro_rules! lit_nil {
    ($span:expr) => {
        firefly_syntax_base::Literal::nil($span)
    };
}
