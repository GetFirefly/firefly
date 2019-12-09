macro_rules! boolean_infix_operator {
    ($left:ident, $right:ident, $operator:tt) => {{
        use std::convert::TryInto;

        use anyhow::*;

        let left_bool: bool = $left.try_into().with_context(|| format!("left ({}) must be a bool", $left))?;
        let right_bool: bool = $right.try_into().with_context(|| format!("right ({}) must be a bool", $right))?;
        let output_bool = left_bool $operator right_bool;

        Ok(output_bool.into())
    }};
}
