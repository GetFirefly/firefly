macro_rules! boolean_infix_operator {
    ($left:ident, $right:ident, $operator:tt) => {{
        let (left_bool, right_bool) = lumen_rt_core::context::terms_try_into_bools(stringify!($left), $left, stringify!($right), $right)?;
        let output_bool = left_bool $operator right_bool;

        Ok(output_bool.into())
    }};
}
