macro_rules! boolean_infix_operator {
    ($left:ident, $right:ident, $operator:tt) => {{
        let left_bool: bool = $left.try_into()?;
        let right_bool: bool = $right.try_into()?;
        let output_bool = left_bool $operator right_bool;

        Ok(output_bool.into())
    }};
}
